"""
TinyVLM Main Model
Integrates DeiT-Tiny, BitNet, Multimodal Adapter, Episodic Memory, and ScopeNet
"""

import torch
import torch.nn as nn
import json
import os
from typing import Optional, Tuple, List

from deit_encoder import DeiTTinyEncoder, ImagePreprocessor
from bitnet_model import BitNetModel, BitNetConfig
from multimodal_adapter import MultimodalAdapter
from episodic_memory import EpisodicMemory
from scope_net import ScopeNet
from attention_visualizer import AttentionVisualizer
from image_text_alignment import ImageTextAlignmentModule, CrossModalFusionModule


class TinyVLM(nn.Module):
    """
    Tiny Vision-Language Model

    Architecture:
        1. DeiT-Tiny vision encoder extracts patch embeddings
        2. Multimodal adapter projects and pools to prefix tokens
        3. Prefix tokens inserted into BitNet sequence
        4. Episodic memory provides additional context via KV injection
        5. ScopeNet decides when to apply memory
        6. BitNet processes fused sequence

    Total size target: < 500 MB
    """

    def __init__(self, config_path=None, device='cuda'):
        super().__init__()

        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(
                __file__), "..", "configs", "model_config.json")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.device = device

        # Initialize components
        print("Initializing TinyVLM components...")

        # Vision encoder
        self.vision_encoder = DeiTTinyEncoder(
            config_dict=self.config['deit_tiny'],
            pretrained=True
        )

        # Image preprocessor
        self.image_preprocessor = ImagePreprocessor(
            image_size=self.config['deit_tiny']['image_size']
        )

        # Multimodal adapter
        self.adapter = MultimodalAdapter(config_dict=self.config['adapter'])

        # Language model (BitNet)
        bitnet_config_dict = self.config['bitnet']
        bitnet_config = BitNetConfig(
            hidden_size=bitnet_config_dict['hidden_size'],
            num_layers=bitnet_config_dict['num_layers'],
            num_attention_heads=bitnet_config_dict['num_attention_heads'],
            num_key_value_heads=bitnet_config_dict['num_key_value_heads'],
            head_dim=bitnet_config_dict['head_dim'],
            intermediate_size=bitnet_config_dict['intermediate_size'],
            vocab_size=bitnet_config_dict['vocab_size'],
            max_position_embeddings=bitnet_config_dict['max_position_embeddings'],
            rope_theta=bitnet_config_dict['rope_theta'],
            rms_norm_eps=bitnet_config_dict['rms_norm_eps'],
            tie_word_embeddings=bitnet_config_dict['tie_word_embeddings']
        )
        self.language_model = BitNetModel(bitnet_config)

        # Episodic memory
        self.episodic_memory = EpisodicMemory(
            config_dict={
                'memory': self.config['memory'], 'bitnet': self.config['bitnet']},
            device=device
        )

        # Scope detector
        self.scope_net = ScopeNet(config_dict=self.config['scope'])

        # Attention visualizer (for monitoring)
        self.attention_visualizer = AttentionVisualizer(
            config_dict=self.config['attention_viz'])

        # EVO-1 Image-Text Alignment (contrastive learning)
        self.image_text_alignment = ImageTextAlignmentModule(
            image_dim=self.config['deit_tiny']['embed_dim'],
            text_dim=self.config['bitnet']['hidden_size'],
            projection_dim=self.config['alignment'].get('projection_dim', 512),
            temperature=self.config['alignment'].get('temperature', 0.07)
        )

        # Cross-modal fusion (following EVO-1 methodology)
        self.cross_modal_fusion = CrossModalFusionModule(
            image_dim=self.config['deit_tiny']['embed_dim'],
            text_dim=self.config['bitnet']['hidden_size'],
            num_heads=self.config['alignment'].get('fusion_heads', 8),
            num_layers=self.config['alignment'].get('fusion_layers', 2)
        )

        # Special tokens
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

        # Memory state
        self.memory_state = None

        print("TinyVLM initialized successfully")

    def _detach_memory_state(self, memory_state):
        """Detach tensors inside memory state to avoid graph retention"""
        if memory_state is None:
            return None
        detached = []
        for tensor in memory_state:
            if torch.is_tensor(tensor):
                detached.append(tensor.detach())
            else:
                detached.append(tensor)
        return tuple(detached)

    def encode_images(self, images):
        """
        Encode images to prefix tokens

        Args:
            images: (batch, 3, H, W) or list of images

        Returns:
            prefix_tokens: (batch, K_prefix, bitnet_hidden_dim)
        """
        # Preprocess images
        if isinstance(images, list):
            images = torch.stack(images)

        images = self.image_preprocessor(images).to(self.device)

        # Extract patch embeddings
        patch_embeddings = self.vision_encoder(images)

        # Project to prefix tokens
        prefix_tokens = self.adapter(patch_embeddings)

        return prefix_tokens

    def build_fused_sequence(self, prefix_tokens, text_token_ids):
        """
        Build fused sequence: [BOS] + prefix_tokens + text_tokens + [EOS]

        Args:
            prefix_tokens: (batch, K_prefix, hidden_dim)
            text_token_ids: (batch, text_len) token IDs

        Returns:
            fused_embeds: (batch, total_len, hidden_dim)
            image_token_range: (start_idx, end_idx) for prefix tokens
            text_token_range: (start_idx, end_idx) for text tokens
        """
        batch_size = prefix_tokens.size(0)

        # Get text embeddings
        text_embeds = self.language_model.tok_embeddings(text_token_ids)

        # Get special token embeddings
        bos_embeds = self.language_model.tok_embeddings(
            torch.full((batch_size, 1), self.bos_token_id, device=self.device)
        )
        eos_embeds = self.language_model.tok_embeddings(
            torch.full((batch_size, 1), self.eos_token_id, device=self.device)
        )

        # Concatenate: [BOS] + prefix_tokens + text_tokens + [EOS]
        fused_embeds = torch.cat(
            [bos_embeds, prefix_tokens, text_embeds, eos_embeds], dim=1)

        # Track token ranges
        image_token_range = (1, 1 + prefix_tokens.size(1))
        text_token_range = (1 + prefix_tokens.size(1), 1 +
                            prefix_tokens.size(1) + text_token_ids.size(1))

        return fused_embeds, image_token_range, text_token_range

    def compute_alignment_loss(self, image_embeds, text_embeds):
        """
        Compute EVO-1 style contrastive image-text alignment loss

        Args:
            image_embeds: (batch, num_patches, embed_dim) precomputed image embeddings
            text_embeds: (batch, text_len, hidden_dim) precomputed text embeddings

        Returns:
            alignment_loss: contrastive loss between image and text
            similarity_matrix: image-text similarity
        """
        # Compute contrastive alignment loss (EVO-1 methodology)
        alignment_loss, similarity = self.image_text_alignment(
            image_embeds,
            text_embeds,
            return_similarity=True
        )

        return alignment_loss, similarity

    def forward(self, images, text_token_ids, use_memory=True, use_fusion=False, return_attention=False):
        """
        Forward pass

        Args:
            images: (batch, 3, H, W) images
            text_token_ids: (batch, text_len) text tokens
            use_memory: whether to use episodic memory
            use_fusion: whether to use cross-modal fusion (EVO-1 style)
            return_attention: whether to return attention weights

        Returns:
            logits: (batch, seq_len, vocab_size)
            memory_state: updated memory state
            attention_weights: optional attention weights
            cached_embeds: dict with 'image_embeds' and 'text_embeds' for loss computation
        """
        batch_size = images.size(0)

        # Preprocess and encode images once
        images_preprocessed = self.image_preprocessor(images).to(self.device)
        image_embeds = self.vision_encoder(images_preprocessed)  # (batch, num_patches, embed_dim)
        
        # Project to prefix tokens
        prefix_tokens = self.adapter(image_embeds)

        # Get text embeddings
        text_embeds = self.language_model.tok_embeddings(text_token_ids)
        
        # Optional: Apply cross-modal fusion (EVO-1 methodology)
        if use_fusion:
            # Apply cross-attention fusion
            text_embeds_fused = self.cross_modal_fusion(image_embeds, text_embeds)
            
            # Build sequence with fused text embeddings
            bos_embeds = self.language_model.tok_embeddings(
                torch.full((batch_size, 1), self.bos_token_id, device=self.device)
            )
            eos_embeds = self.language_model.tok_embeddings(
                torch.full((batch_size, 1), self.eos_token_id, device=self.device)
            )
            
            fused_embeds = torch.cat([bos_embeds, prefix_tokens, text_embeds_fused, eos_embeds], dim=1)
            image_token_range = (1, 1 + prefix_tokens.size(1))
            text_token_range = (1 + prefix_tokens.size(1), 1 + prefix_tokens.size(1) + text_token_ids.size(1))
        else:
            # Build fused sequence (simple concatenation)
            fused_embeds, image_token_range, text_token_range = self.build_fused_sequence(
                prefix_tokens, text_token_ids
            )

        # Get context embedding for scope decision (use mean of prefix tokens)
        context_embed = prefix_tokens.mean(dim=1)  # (batch, hidden_dim)

        # Decide whether to use memory
        if use_memory:
            scope_decision, scope_prob = self.scope_net(context_embed)
        else:
            scope_decision = torch.zeros(batch_size, device=self.device)
            scope_prob = torch.zeros(batch_size, device=self.device)

        # Prepare memory KVs if needed
        memory_kvs = None
        if use_memory and scope_decision.sum() > 0:
            # Read from memory
            if self.memory_state is None:
                # Initialize memory
                self.memory_state, _ = self.episodic_memory.write(
                    context_embed.unsqueeze(0),
                    batch_size=batch_size
                )
                self.memory_state = self._detach_memory_state(self.memory_state)

            # Read memory
            z_retrieved, Z_r_kv, dkl_w = self.episodic_memory.read(
                context_embed,
                self.memory_state
            )

            # Inject memory KVs for each layer
            memory_kvs = []
            for layer_idx in range(self.config['bitnet']['num_layers']):
                mem_k, mem_v = self.episodic_memory.inject_to_kv_cache(
                    Z_r_kv, layer_idx)

                # Apply scope decision (mask for batch items that don't use memory)
                scope_mask = scope_decision.view(-1, 1, 1, 1)
                mem_k = mem_k * scope_mask
                mem_v = mem_v * scope_mask

                memory_kvs.append((mem_k, mem_v))

        # Forward through BitNet with memory injection
        logits, kv_caches, attention_weights = self.language_model(
            inputs_embeds=fused_embeds,
            memory_kvs=memory_kvs,
            return_attention=return_attention
        )

        return logits, self.memory_state, attention_weights, {
            'scope_decision': scope_decision,
            'scope_prob': scope_prob,
            'image_token_range': image_token_range,
            'text_token_range': text_token_range,
            'cached_embeds': {
                'image_embeds': image_embeds,
                'text_embeds': text_embeds
            }
        }

    def update_memory(self, images, text_token_ids):
        """
        Update episodic memory with new experience

        Args:
            images: (batch, 3, H, W)
            text_token_ids: (batch, text_len)

        Returns:
            dkl_M: memory KL divergence
        """
        # Encode images
        prefix_tokens = self.encode_images(images)
        context_embed = prefix_tokens.mean(dim=1)

        # Write to memory
        self.memory_state, dkl_M = self.episodic_memory.write(
            context_embed.unsqueeze(0),
            batch_size=images.size(0)
        )
        self.memory_state = self._detach_memory_state(self.memory_state)

        return dkl_M

    def reset_memory(self):
        """Reset episodic memory state"""
        self.memory_state = None

    def freeze_vision_encoder(self, num_stages=8):
        """Freeze early vision encoder layers"""
        self.vision_encoder.freeze_stages(num_stages)

    def freeze_language_model(self, num_layers=None):
        """Freeze language model layers (except last N)"""
        if num_layers is None:
            num_layers = self.config['bitnet']['num_layers']

        # Freeze embeddings
        for param in self.language_model.tok_embeddings.parameters():
            param.requires_grad = False

        # Freeze layers
        for i in range(num_layers):
            for param in self.language_model.layers[i].parameters():
                param.requires_grad = False

    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer"""
        return [p for p in self.parameters() if p.requires_grad]

    def save_checkpoint(self, path):
        """Save full model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'memory_state': self.memory_state
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load full model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.memory_state = checkpoint.get('memory_state', None)
        print(f"Checkpoint loaded from {path}")

    def estimate_model_size(self):
        """Estimate model size in MB"""
        param_size = sum(p.numel() * p.element_size()
                         for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        total_size_mb = (param_size + buffer_size) / (1024 ** 2)
        return total_size_mb




    print("\nTinyVLM test passed!")
