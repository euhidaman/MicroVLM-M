from src.wandb_counter import WandBRunCounter
from src.dataset import CC12MDataset, collate_fn
from src.tiny_vlm import TinyVLM
import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")


class Stage2Trainer:
    def __init__(self, config_path, checkpoint_path=None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.device = torch.device(self.config['training']['device'])
        self.use_wandb = self.config['logging']['use_wandb'] and WANDB_AVAILABLE

        self.model = TinyVLM(
            config_path=self.config['model_config_path'],
            device=self.device
        )

        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.model.load_checkpoint(checkpoint_path)
        elif 'stage1_checkpoint' in self.config:
            print(
                f"Loading Stage 1 checkpoint from {self.config['stage1_checkpoint']}")
            self.model.load_checkpoint(self.config['stage1_checkpoint'])
        else:
            raise ValueError("No checkpoint provided for Stage 2 training")

        self._setup_freezing()
        self._setup_optimizer()
        self._setup_data()

        if self.use_wandb:
            counter = WandBRunCounter()
            run_name, run_number = counter.get_next_run_name('stage2')
            wandb.init(
                project=self.config['logging']['wandb_project'],
                entity=self.config['logging']['wandb_entity'],
                name=run_name,
                config=self.config
            )
            print(f"WandB run initialized: {run_name} (Run #{run_number})")

        self.scaler = GradScaler(enabled=self.config['training']['use_amp'])
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _setup_freezing(self):
        freeze_vision = self.config['training']['freeze_vision_stages']
        freeze_lm = self.config['training']['freeze_lm_layers']
        unfreeze_last_n = self.config['training'].get(
            'unfreeze_last_n_lm_layers', 4)

        self.model.freeze_vision_encoder(num_stages=freeze_vision)
        self.model.freeze_language_model(num_layers=freeze_lm)

        print(f"Frozen vision encoder: {freeze_vision} stages")
        print(
            f"Frozen LM layers: {freeze_lm}/{self.model.language_model.config['num_layers']}")

        total_layers = self.model.language_model.config['num_layers']
        unfreeze_start = total_layers - unfreeze_last_n

        for i in range(unfreeze_start, total_layers):
            for param in self.model.language_model.layers[i].parameters():
                param.requires_grad = True

        print(
            f"Unfrozen last {unfreeze_last_n} LM layers (layers {unfreeze_start}-{total_layers-1})")

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    def _setup_optimizer(self):
        opt_config = self.config['optimization']
        train_config = self.config['training']

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            betas=(opt_config['beta1'], opt_config['beta2']),
            eps=opt_config['eps']
        )

        if opt_config['lr_scheduler'] == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['max_steps'],
                eta_min=opt_config['min_lr']
            )
        else:
            self.scheduler = None

    def _setup_data(self):
        data_config = self.config['data']
        train_config = self.config['training']

        self.train_dataset = CC12MDataset(
            metadata_path=data_config['train_metadata'],
            image_dir=data_config['image_dir'],
            image_size=224
        )

        self.val_dataset = CC12MDataset(
            metadata_path=data_config['val_metadata'],
            image_dir=data_config['image_dir'],
            image_size=224
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=data_config.get('prefetch_factor', 2)
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def compute_loss(self, batch):
        images, text_tokens, attention_mask = batch
        images = images.to(self.device)
        text_tokens = text_tokens.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with autocast(enabled=self.config['training']['use_amp']):
            logits, memory_state, attn_weights, metadata = self.model(
                images,
                text_tokens,
                use_memory=True,
                return_attention=True
            )

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = text_tokens[:, 1:].contiguous()

            lm_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0
            )

            if memory_state is not None and 'memory_vector' in memory_state:
                Z_r = memory_state['memory_vector']
                target = metadata.get('fused_context', torch.zeros_like(Z_r))
                memory_loss = nn.functional.mse_loss(Z_r, target)
            else:
                memory_loss = torch.tensor(0.0, device=self.device)

            scope_decision = metadata.get('scope_decision', 0.5)
            scope_target = torch.ones_like(scope_decision) * 0.7
            scope_loss = nn.functional.mse_loss(scope_decision, scope_target)

            weights = self.config['loss_weights']
            total_loss = (
                weights['lm_loss'] * lm_loss +
                weights['memory_loss'] * memory_loss +
                weights['scope_loss'] * scope_loss
            )

        return total_loss, lm_loss, memory_loss, scope_loss, scope_decision.mean()

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        total_loss, lm_loss, memory_loss, scope_loss, scope_decision = self.compute_loss(
            batch)

        self.scaler.scale(total_loss).backward()

        if self.config['training']['grad_clip'] > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip']
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'memory_loss': memory_loss.item(),
            'scope_loss': scope_loss.item(),
            'scope_decision': scope_decision.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            loss, _, _, _, _ = self.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def save_checkpoint(self, path, is_best=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

        if is_best:
            best_path = os.path.join(
                self.config['checkpointing']['checkpoint_dir'],
                'best',
                'checkpoint.pt'
            )
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint saved: {best_path}")

    def train(self):
        train_config = self.config['training']
        log_config = self.config['logging']

        num_epochs = train_config['num_epochs']
        max_steps = train_config['max_steps']

        print(f"\nStarting Stage 2 Training")
        print(f"Epochs: {num_epochs}, Max steps: {max_steps}")
        print(f"Device: {self.device}")

        for epoch in range(num_epochs):
            epoch_pbar = tqdm(self.train_loader,
                              desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in epoch_pbar:
                metrics = self.train_step(batch)
                self.global_step += 1

                if self.global_step % log_config['log_interval'] == 0:
                    epoch_pbar.set_postfix({
                        'loss': f"{metrics['total_loss']:.4f}",
                        'lm': f"{metrics['lm_loss']:.4f}",
                        'mem': f"{metrics['memory_loss']:.4f}",
                        'lr': f"{metrics['lr']:.2e}"
                    })

                    if self.use_wandb:
                        wandb.log({
                            'train/total_loss': metrics['total_loss'],
                            'train/lm_loss': metrics['lm_loss'],
                            'train/memory_loss': metrics['memory_loss'],
                            'train/scope_loss': metrics['scope_loss'],
                            'train/scope_decision': metrics['scope_decision'],
                            'train/learning_rate': metrics['lr'],
                            'global_step': self.global_step
                        })

                if self.global_step % log_config['val_interval'] == 0:
                    val_loss = self.validate()
                    print(
                        f"\nStep {self.global_step} - Validation Loss: {val_loss:.4f}")

                    if self.use_wandb:
                        wandb.log({
                            'val/loss': val_loss,
                            'global_step': self.global_step
                        })

                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        print(f"New best validation loss: {val_loss:.4f}")

                    checkpoint_path = os.path.join(
                        self.config['checkpointing']['checkpoint_dir'],
                        f'step_{self.global_step}',
                        'checkpoint.pt'
                    )
                    self.save_checkpoint(checkpoint_path, is_best=is_best)

                if self.global_step % log_config['save_interval'] == 0:
                    checkpoint_path = os.path.join(
                        self.config['checkpointing']['checkpoint_dir'],
                        f'step_{self.global_step}',
                        'checkpoint.pt'
                    )
                    self.save_checkpoint(checkpoint_path)

                if self.global_step >= max_steps:
                    print(f"\nReached max steps: {max_steps}")
                    break

            if self.global_step >= max_steps:
                break

        final_path = os.path.join(
            self.config['checkpointing']['checkpoint_dir'],
            'final',
            'checkpoint.pt'
        )
        self.save_checkpoint(final_path)

        if self.use_wandb:
            wandb.finish()

        print(f"\nStage 2 Training completed!")
        print(f"Total steps: {self.global_step}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Stage 2 Training for MicroVLM-M')
    parser.add_argument('--config', type=str, default='configs/stage2_config.json',
                        help='Path to stage 2 config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to Stage 1 checkpoint (overrides config)')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Creating default config...")

        default_config = {
            "model_config_path": "configs/model_config.json",
            "stage1_checkpoint": "checkpoints/stage1/best/checkpoint.pt",
            "data": {
                "train_metadata": "data/cc12m/train_metadata.json",
                "val_metadata": "data/cc12m/val_metadata.json",
                "image_dir": "data/cc12m/images",
                "num_workers": 4,
                "prefetch_factor": 2
            },
            "training": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "batch_size": 16,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "grad_clip": 1.0,
                "num_epochs": 3,
                "max_steps": 10000,
                "warmup_steps": 500,
                "use_amp": True,
                "freeze_vision_stages": 12,
                "freeze_lm_layers": 26,
                "unfreeze_last_n_lm_layers": 4
            },
            "loss_weights": {
                "lm_loss": 1.0,
                "memory_loss": 0.05,
                "scope_loss": 0.005
            },
            "optimization": {
                "optimizer": "adamw",
                "lr_scheduler": "cosine",
                "min_lr": 1e-7,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8
            },
            "logging": {
                "log_interval": 100,
                "val_interval": 500,
                "save_interval": 2000,
                "visualize_interval": 2000,
                "use_wandb": True,
                "wandb_entity": "aman-derax20",
                "wandb_project": "MicroVLM-M"
            },
            "checkpointing": {
                "checkpoint_dir": "checkpoints/stage2",
                "save_best": True,
                "save_last": True,
                "keep_n_checkpoints": 3
            }
        }

        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Default config created: {args.config}")

    trainer = Stage2Trainer(args.config, args.checkpoint)
    trainer.train()


if __name__ == '__main__':
    main()
