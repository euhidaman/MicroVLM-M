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


class Stage1Trainer:
    """
    Stage 1 Trainer: Train adapters, memory, and scope net
    """

    def __init__(self, config):
        self.config = config
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
        self.train_dataset = CC12MDataset(
            metadata_path=config['data']['train_metadata'],
            image_size=224
        )

        self.val_dataset = CC12MDataset(
            metadata_path=config['data']['val_metadata'],
            image_size=224
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
        alignment_loss, similarity_matrix = self.model.compute_alignment_loss(
            images,
            token_ids
        )

        # Memory reconstruction loss (encourage meaningful memory)
        if memory_state is not None:
            memory_mean = memory_state[0]  # (batch, k_mem, c_mem)

            # Reconstruct input from memory
            # Simple reconstruction: mean of memory should approximate input embedding
            prefix_embeds = self.model.encode_images(images)
            target_embed = prefix_embeds.mean(dim=1)  # (batch, hidden_dim)
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


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Training (EVO-1 Alignment + Larimar Memory)")
    parser.add_argument('--config', type=str, default='configs/stage1_config.json',
                        help='Training configuration file')
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Please create a config file or use the provided configs/stage1_config.json")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Train
    trainer = Stage1Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
