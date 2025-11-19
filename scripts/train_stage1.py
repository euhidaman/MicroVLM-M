"""
Stage 1 Training: Adapter, Memory, and ScopeNet
Freezes BitNet and DeiT early layers, trains only adapters and memory components
"""

import os
import sys
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wandb_counter import WandBRunCounter
from dataset import CC12MDataset
from tiny_vlm import TinyVLM
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import json
from huggingface_hub import HfApi, create_repo
import glob


class Stage1Trainer:
    """
    Stage 1 Trainer: Train adapters, memory, and scope net
    """

    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = torch.device(config['training']['device'])

        # Initialize model
        print("Initializing TinyVLM...")
        self.model = TinyVLM(
            config_path=config['model_config_path'],
            device=self.device
        ).to(self.device)

        # Freeze components
        print("Freezing vision encoder and language model...")
        self.model.freeze_vision_encoder(
            num_stages=config['training']['freeze_vision_stages'])
        self.model.freeze_language_model(num_layers=config['training']['freeze_lm_layers'])

        # Print trainable parameters
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

        # Loss functions
        self.lm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.alignment_loss_fn = nn.MSELoss()

        # Optimizer
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(config['optimization']['beta1'], config['optimization']['beta2']),
            eps=config['optimization']['eps']
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['max_steps'],
            eta_min=config['optimization']['min_lr']
        )

        # Dataset
        print("Loading dataset...")
        if args.small_scale:
            print("SMALL-SCALE MODE: Training on 1000 randomly sampled image-caption pairs")
        
        self.train_dataset = CC12MDataset(
            metadata_path=config['data']['train_metadata'],
            image_size=224,
            small_scale=args.small_scale
        )

        self.val_dataset = CC12MDataset(
            metadata_path=config['data']['val_metadata'],
            image_size=224,
            small_scale=args.small_scale
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            collate_fn=CC12MDataset.collate_fn,
            pin_memory=True,
            prefetch_factor=config['data'].get('prefetch_factor', 2)
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            collate_fn=CC12MDataset.collate_fn,
            pin_memory=True,
            prefetch_factor=config['data'].get('prefetch_factor', 2)
        )

        # WandB
        if config['logging']['use_wandb']:
            try:
                counter = WandBRunCounter(project_name=config['logging']['wandb_project'])
                run_name, run_number = counter.get_next_run_name('stage1')

                wandb.init(
                    project=config['logging']['wandb_project'],
                    name=run_name,
                    config=config,
                    entity=config['logging'].get('wandb_entity', None)  # None = use default
                )
                print(f"WandB run: {run_name} (#{run_number})")
            except Exception as e:
                print(f"Warning: WandB initialization failed: {e}")
                print("Continuing training without WandB logging...")
                config['logging']['use_wandb'] = False

        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Hugging Face Hub setup
        self.hf_repo_id = config.get('huggingface', {}).get('repo_id', 'euhidaman/MicroVLM-M')
        self.hf_upload = config.get('huggingface', {}).get('upload_enabled', True)
        self.max_local_checkpoints = 3
        
        # Create HF repo if needed
        if self.hf_upload:
            try:
                self.hf_api = HfApi()
                create_repo(self.hf_repo_id, exist_ok=True, private=False)
                print(f"Hugging Face repo: {self.hf_repo_id}")
            except Exception as e:
                print(f"Warning: HF Hub setup failed: {e}")
                self.hf_upload = False

    def compute_loss(self, batch):
        """
        Compute training loss with EVO-1 style alignment

        Returns:
            total_loss, loss_dict
        """
        images = batch['images'].to(self.device)
        token_ids = batch['token_ids'].to(self.device)

        # Forward pass with EVO-1 alignment
        logits, memory_state, attn_weights, metadata = self.model(
            images,
            token_ids[:, :-1],  # Input tokens (exclude last)
            use_memory=True,
            use_fusion=self.config.get('use_cross_modal_fusion', False),
            return_attention=False
        )

        # Language modeling loss
        # Shift logits and targets for next-token prediction
        vocab_size = logits.size(-1)
        # Last text_len tokens
        lm_logits = logits[:, -token_ids.size(1)+1:, :].contiguous()
        lm_targets = token_ids[:, 1:].contiguous()  # Targets (exclude first)

        lm_logits_flat = lm_logits.view(-1, vocab_size)
        lm_targets_flat = lm_targets.view(-1)

        lm_loss = self.lm_loss_fn(lm_logits_flat, lm_targets_flat)

        # EVO-1 Image-Text Alignment Loss (contrastive learning)
        # Use cached embeddings from forward pass to avoid recomputation
        cached_embeds = metadata.get('cached_embeds', {})
        if cached_embeds:
            alignment_loss, similarity_matrix = self.model.compute_alignment_loss(
                cached_embeds['image_embeds'],
                cached_embeds['text_embeds']
            )
        else:
            # Fallback if cached_embeds not available
            alignment_loss = torch.tensor(0.0, device=self.device)

        # Memory reconstruction loss (encourage meaningful memory)
        if memory_state is not None:
            memory_mean = memory_state[0]  # (batch, k_mem, c_mem)

            # Reconstruct input from memory
            # Simple reconstruction: mean of memory should approximate input embedding
            with torch.no_grad():
                prefix_embeds = self.model.encode_images(images)
                target_embed = prefix_embeds.mean(dim=1).detach()  # (batch, hidden_dim)
            recon_embed = memory_mean.mean(dim=1)  # (batch, c_mem)

            memory_loss = self.alignment_loss_fn(recon_embed, target_embed)
        else:
            memory_loss = torch.tensor(0.0, device=self.device)

        # Scope loss (encourage balanced scope decisions)
        scope_prob = metadata['scope_prob']
        # Encourage exploration
        scope_loss = torch.mean((scope_prob - 0.5) ** 2)

        # Total loss (including EVO-1 alignment)
        total_loss = (
            self.config['loss_weights']['lm_loss'] * lm_loss +
            self.config['loss_weights'].get('alignment_loss', 0.5) * alignment_loss +
            self.config['loss_weights']['memory_loss'] * memory_loss +
            self.config['loss_weights']['scope_loss'] * scope_loss
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'memory_loss': memory_loss.item(),
            'scope_loss': scope_loss.item(),
            'scope_decision': metadata['scope_decision'].mean().item()
        }

        return total_loss, loss_dict

    def train_step(self, batch):
        """Single training step"""
        self.model.train()

        loss, loss_dict = self.compute_loss(batch)

        loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
            # Gradient clipping
            if self.config['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return loss_dict

    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                _, loss_dict = self.compute_loss(batch)
                total_loss += loss_dict['total_loss']
                num_batches += 1

                if num_batches >= 100:  # Validation batches
                    break

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self):
        """Main training loop"""
        print("Starting Stage 1 training (EVO-1 alignment + Larimar memory)...")

        for epoch in range(self.config['training']['num_epochs']):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            epoch_start_step = self.global_step

            for batch in pbar:
                loss_dict = self.train_step(batch)

                self.global_step += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'lm': f"{loss_dict['lm_loss']:.4f}",
                    'align': f"{loss_dict['alignment_loss']:.4f}",
                    'mem': f"{loss_dict['memory_loss']:.4f}"
                })

                # Log to WandB
                if self.config['logging']['use_wandb'] and self.global_step % self.config['logging']['log_interval'] == 0:
                    wandb.log({
                        'train/total_loss': loss_dict['total_loss'],
                        'train/lm_loss': loss_dict['lm_loss'],
                        'train/alignment_loss': loss_dict['alignment_loss'],
                        'train/memory_loss': loss_dict['memory_loss'],
                        'train/scope_loss': loss_dict['scope_loss'],
                        'train/scope_decision': loss_dict['scope_decision'],
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    })

                # Validation
                if self.global_step % self.config['logging']['val_interval'] == 0:
                    val_loss = self.validate()
                    print(f"\nValidation loss: {val_loss:.4f}")

                    if self.config['logging']['use_wandb']:
                        wandb.log({
                            'val/loss': val_loss,
                            'global_step': self.global_step
                        })

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best')

                # Save checkpoint
                if self.global_step % self.config['logging']['save_interval'] == 0:
                    self.save_checkpoint(f'step_{self.global_step}')

                # Max steps
                if self.global_step >= self.config['training']['max_steps']:
                    print("Reached max steps")
                    return
            
            # End of epoch - save and upload
            print(f"\nEpoch {epoch+1} complete. Saving checkpoint...")
            checkpoint_name = f"epoch_{epoch+1}"
            self.save_checkpoint(checkpoint_name)
            
            # Upload to Hugging Face (overwrites previous)
            if self.hf_upload:
                self.upload_to_hf(checkpoint_name)
            
            # Clean up old local checkpoints
            self.cleanup_old_checkpoints()

        print("Training complete!")

    def save_checkpoint(self, name):
        """Save model checkpoint"""
        save_dir = os.path.join(self.config['checkpointing']['checkpoint_dir'], name)
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        return save_dir
    
    def upload_to_hf(self, checkpoint_name):
        """Upload checkpoint to Hugging Face Hub (overwrites previous)"""
        try:
            checkpoint_dir = os.path.join(self.config['checkpointing']['checkpoint_dir'], checkpoint_name)
            print(f"Uploading {checkpoint_name} to Hugging Face...")
            
            # Generate README with current training stats
            self._generate_model_card(checkpoint_dir)
            
            self.hf_api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=self.hf_repo_id,
                repo_type="model",
                commit_message=f"Update model - {checkpoint_name} (step {self.global_step})"
            )
            print(f"Successfully uploaded to https://huggingface.co/{self.hf_repo_id}")
        except Exception as e:
            print(f"Warning: Failed to upload to HF Hub: {e}")
    
    def _generate_model_card(self, checkpoint_dir):
        """Generate README.md with current training stats"""
        from datetime import datetime
        
        # Read template
        template_path = os.path.join(os.path.dirname(__file__), '..', 'model_card_template.md')
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Fill in current values
        readme_content = template.replace(
            '{timestamp}', datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        ).replace(
            '{global_step}', str(self.global_step)
        ).replace(
            '{val_loss:.4f}', f'{self.best_val_loss:.4f}'
        )
        
        # Save to checkpoint directory
        readme_path = os.path.join(checkpoint_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"Generated model card: {readme_path}")
    
    def cleanup_old_checkpoints(self):
        """Keep only the N most recent epoch checkpoints locally"""
        checkpoint_base = self.config['checkpointing']['checkpoint_dir']
        epoch_dirs = glob.glob(os.path.join(checkpoint_base, 'epoch_*'))
        
        # Sort by modification time (newest first)
        epoch_dirs.sort(key=os.path.getmtime, reverse=True)
        
        # Remove old checkpoints beyond max_local_checkpoints
        for old_dir in epoch_dirs[self.max_local_checkpoints:]:
            try:
                import shutil
                shutil.rmtree(old_dir)
                print(f"Removed old checkpoint: {old_dir}")
            except Exception as e:
                print(f"Warning: Failed to remove {old_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Training (EVO-1 Alignment + Larimar Memory)")
    parser.add_argument('--config', type=str, default='configs/stage1_config.json',
                        help='Training configuration file')
    parser.add_argument('--small-scale', action='store_true',
                        help='Use small-scale mode (1000 random samples) for quick validation')
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Please create a config file or use the provided configs/stage1_config.json")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Train
    trainer = Stage1Trainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
