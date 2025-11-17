"""
Stage 1 Training: Adapter, Memory, and ScopeNet
Freezes BitNet and DeiT early layers, trains only adapters and memory components
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tiny_vlm import TinyVLM
from dataset import CC12MDataset
from wandb_counter import WandBRunCounter

class Stage1Trainer:
    """
    Stage 1 Trainer: Train adapters, memory, and scope net
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Initialize model
        print("Initializing TinyVLM...")
        self.model = TinyVLM(
            config_path=config['model_config_path'],
            device=self.device
        ).to(self.device)
        
        # Freeze components
        print("Freezing vision encoder and language model...")
        self.model.freeze_vision_encoder(num_stages=config['freeze_vision_stages'])
        self.model.freeze_language_model(num_layers=config['freeze_lm_layers'])
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # Loss functions
        self.lm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.alignment_loss_fn = nn.MSELoss()
        
        # Optimizer
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['max_steps'],
            eta_min=config['learning_rate'] * 0.1
        )
        
        # Dataset
        print("Loading dataset...")
        self.train_dataset = CC12MDataset(
            metadata_path=config['train_metadata_path'],
            image_size=224
        )
        
        self.val_dataset = CC12MDataset(
            metadata_path=config['val_metadata_path'],
            image_size=224
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=CC12MDataset.collate_fn,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=CC12MDataset.collate_fn,
            pin_memory=True
        )
        
        # WandB
        if config['use_wandb']:
            counter = WandBRunCounter(project_name='MicroVLM-M')
            run_name, run_number = counter.get_next_run_name('stage1')
            
            wandb.init(
                project='MicroVLM-M',
                name=run_name,
                config=config,
                entity=config['wandb_entity']
            )
            print(f"WandB run: {run_name} (#{run_number})")
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def compute_loss(self, batch):
        """
        Compute training loss
        
        Returns:
            total_loss, loss_dict
        """
        images = batch['images'].to(self.device)
        token_ids = batch['token_ids'].to(self.device)
        
        # Forward pass
        logits, memory_state, attn_weights, metadata = self.model(
            images,
            token_ids[:, :-1],  # Input tokens (exclude last)
            use_memory=True,
            return_attention=False
        )
        
        # Language modeling loss
        # Shift logits and targets for next-token prediction
        vocab_size = logits.size(-1)
        lm_logits = logits[:, -token_ids.size(1)+1:, :].contiguous()  # Last text_len tokens
        lm_targets = token_ids[:, 1:].contiguous()  # Targets (exclude first)
        
        lm_logits_flat = lm_logits.view(-1, vocab_size)
        lm_targets_flat = lm_targets.view(-1)
        
        lm_loss = self.lm_loss_fn(lm_logits_flat, lm_targets_flat)
        
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
        scope_loss = torch.mean((scope_prob - 0.5) ** 2)  # Encourage exploration
        
        # Total loss
        total_loss = (
            self.config['lm_loss_weight'] * lm_loss +
            self.config['memory_loss_weight'] * memory_loss +
            self.config['scope_loss_weight'] * scope_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'memory_loss': memory_loss.item(),
            'scope_loss': scope_loss.item(),
            'scope_decision': metadata['scope_decision'].mean().item()
        }
        
        return total_loss, loss_dict
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, loss_dict = self.compute_loss(batch)
        
        loss.backward()
        
        # Gradient clipping
        if self.config['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
        
        self.optimizer.step()
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
                
                if num_batches >= self.config['val_batches']:
                    break
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Main training loop"""
        print("Starting Stage 1 training...")
        
        for epoch in range(self.config['num_epochs']):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            
            for batch in pbar:
                loss_dict = self.train_step(batch)
                
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'lm': f"{loss_dict['lm_loss']:.4f}",
                    'mem': f"{loss_dict['memory_loss']:.4f}"
                })
                
                # Log to WandB
                if self.config['use_wandb'] and self.global_step % self.config['log_interval'] == 0:
                    wandb.log({
                        'train/total_loss': loss_dict['total_loss'],
                        'train/lm_loss': loss_dict['lm_loss'],
                        'train/memory_loss': loss_dict['memory_loss'],
                        'train/scope_loss': loss_dict['scope_loss'],
                        'train/scope_decision': loss_dict['scope_decision'],
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    })
                
                # Validation
                if self.global_step % self.config['val_interval'] == 0:
                    val_loss = self.validate()
                    print(f"\nValidation loss: {val_loss:.4f}")
                    
                    if self.config['use_wandb']:
                        wandb.log({
                            'val/loss': val_loss,
                            'global_step': self.global_step
                        })
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best')
                
                # Save checkpoint
                if self.global_step % self.config['save_interval'] == 0:
                    self.save_checkpoint(f'step_{self.global_step}')
                
                # Max steps
                if self.global_step >= self.config['max_steps']:
                    print("Reached max steps")
                    return
        
        print("Training complete!")
    
    def save_checkpoint(self, name):
        """Save model checkpoint"""
        save_dir = os.path.join(self.config['checkpoint_dir'], 'stage1', name)
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
    parser = argparse.ArgumentParser(description="Stage 1 Training")
    parser.add_argument('--config', type=str, default='configs/stage1_config.json',
                       help='Training configuration file')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'model_config_path': 'configs/model_config.json',
            'train_metadata_path': 'data/cc12m/train_metadata.json',
            'val_metadata_path': 'data/cc12m/val_metadata.json',
            'checkpoint_dir': 'checkpoints',
            'batch_size': 16,
            'num_workers': 4,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'num_epochs': 10,
            'max_steps': 50000,
            'log_interval': 100,
            'val_interval': 1000,
            'val_batches': 100,
            'save_interval': 5000,
            'freeze_vision_stages': 8,
            'freeze_lm_layers': 26,
            'lm_loss_weight': 1.0,
            'memory_loss_weight': 0.1,
            'scope_loss_weight': 0.01,
            'use_wandb': True,
            'wandb_entity': 'aman-derax20'
        }
        
        # Save default config
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created default config: {args.config}")
    
    # Train
    trainer = Stage1Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
