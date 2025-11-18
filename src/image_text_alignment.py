"""
Image-Text Alignment Module (EVO-1 Methodology)
Implements contrastive learning between image and text embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageTextAlignmentModule(nn.Module):
    """
    EVO-1 Image-Text Alignment via Contrastive Learning
    
    Methodology:
        1. Project image embeddings and text embeddings to shared space
        2. Compute similarity matrix between images and texts
        3. Apply contrastive loss (InfoNCE / CLIP-style)
        4. Learn aligned representations for multimodal fusion
    
    This follows EVO-1's approach of joint vision-language embedding
    rather than simple concatenation-based fusion.
    """

    def __init__(self, 
                 image_dim=192,      # DeiT-Tiny embed_dim
                 text_dim=2560,      # BitNet hidden_dim
                 projection_dim=512, # Shared projection space
                 temperature=0.07):  # Temperature for contrastive loss
        super().__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # Project image embeddings to shared space
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim)
        )
        
        # Project text embeddings to shared space
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim)
        )
        
        # Learnable temperature (like CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / temperature)))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        for module in [self.image_projection, self.text_projection]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, image_embeds, text_embeds, return_similarity=False):
        """
        Compute contrastive alignment loss
        
        Args:
            image_embeds: (batch, K_prefix, image_dim) or (batch, image_dim)
            text_embeds: (batch, seq_len, text_dim) or (batch, text_dim)
            return_similarity: whether to return similarity matrix
        
        Returns:
            loss: contrastive loss
            similarity: optional similarity matrix
        """
        # Pool if necessary
        if image_embeds.dim() == 3:
            image_embeds = image_embeds.mean(dim=1)  # (batch, image_dim)
        if text_embeds.dim() == 3:
            text_embeds = text_embeds.mean(dim=1)  # (batch, text_dim)
        
        batch_size = image_embeds.size(0)
        
        # Project to shared space
        image_features = self.image_projection(image_embeds)  # (batch, projection_dim)
        text_features = self.text_projection(text_embeds)    # (batch, projection_dim)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp().clamp(max=100)
        similarity = logit_scale * torch.matmul(image_features, text_features.t())  # (batch, batch)
        
        # Contrastive loss (InfoNCE / CLIP)
        # Positive pairs are on the diagonal (matching image-text pairs)
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # Image-to-text loss
        loss_i2t = F.cross_entropy(similarity, labels)
        
        # Text-to-image loss
        loss_t2i = F.cross_entropy(similarity.t(), labels)
        
        # Total contrastive loss
        loss = (loss_i2t + loss_t2i) / 2.0
        
        if return_similarity:
            return loss, similarity
        return loss
    
    def get_aligned_features(self, image_embeds, text_embeds):
        """
        Get aligned features in shared space (for inference)
        
        Args:
            image_embeds: (batch, K_prefix, image_dim) or (batch, image_dim)
            text_embeds: (batch, seq_len, text_dim) or (batch, text_dim)
        
        Returns:
            image_features: (batch, projection_dim)
            text_features: (batch, projection_dim)
        """
        # Pool if necessary
        if image_embeds.dim() == 3:
            image_embeds = image_embeds.mean(dim=1)
        if text_embeds.dim() == 3:
            text_embeds = text_embeds.mean(dim=1)
        
        # Project and normalize
        image_features = self.image_projection(image_embeds)
        text_features = self.text_projection(text_embeds)
        
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features
    
    def compute_similarity(self, image_embeds, text_embeds):
        """
        Compute similarity between image and text embeddings
        
        Args:
            image_embeds: (batch, K_prefix, image_dim) or (batch, image_dim)
            text_embeds: (batch, seq_len, text_dim) or (batch, text_dim)
        
        Returns:
            similarity: (batch, batch) similarity matrix
        """
        image_features, text_features = self.get_aligned_features(image_embeds, text_embeds)
        
        logit_scale = self.logit_scale.exp().clamp(max=100)
        similarity = logit_scale * torch.matmul(image_features, text_features.t())
        
        return similarity


class CrossModalFusionModule(nn.Module):
    """
    Cross-modal fusion following EVO-1's methodology
    
    Instead of simple concatenation, this uses cross-attention
    to fuse image and text representations bidirectionally.
    """
    
    def __init__(self, 
                 image_dim=192,
                 text_dim=2560,
                 num_heads=8,
                 num_layers=2):
        super().__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        
        # Project image to text dimension for cross-attention
        self.image_to_text_proj = nn.Linear(image_dim, text_dim)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=text_dim,
                num_heads=num_heads,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(text_dim) for _ in range(num_layers)
        ])
        
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, text_dim * 4),
                nn.GELU(),
                nn.Linear(text_dim * 4, text_dim)
            ) for _ in range(num_layers)
        ])
        
        self.ffn_layer_norms = nn.ModuleList([
            nn.LayerNorm(text_dim) for _ in range(num_layers)
        ])
    
    def forward(self, image_embeds, text_embeds, return_attention=False):
        """
        Cross-modal fusion via cross-attention
        
        Args:
            image_embeds: (batch, K_prefix, image_dim)
            text_embeds: (batch, seq_len, text_dim)
            return_attention: whether to return attention weights
        
        Returns:
            fused_embeds: (batch, seq_len, text_dim) fused representations
            attention_weights: optional list of attention weights
        """
        # Project image embeddings to text dimension
        image_proj = self.image_to_text_proj(image_embeds)  # (batch, K_prefix, text_dim)
        
        fused = text_embeds
        attention_weights = []
        
        for i, (cross_attn, ln1, ffn, ln2) in enumerate(zip(
            self.cross_attention_layers,
            self.layer_norms,
            self.ffn,
            self.ffn_layer_norms
        )):
            # Cross-attention: text attends to image
            attn_output, attn_weight = cross_attn(
                query=fused,
                key=image_proj,
                value=image_proj,
                need_weights=return_attention
            )
            
            fused = ln1(fused + attn_output)
            
            # FFN
            ffn_output = ffn(fused)
            fused = ln2(fused + ffn_output)
            
            if return_attention:
                attention_weights.append(attn_weight)
        
        if return_attention:
            return fused, attention_weights
        return fused


if __name__ == "__main__":
    # Test alignment module
    batch_size = 4
    k_prefix = 25
    seq_len = 50
    
    alignment = ImageTextAlignmentModule(
        image_dim=192,
        text_dim=2560,
        projection_dim=512
    )
    
    image_embeds = torch.randn(batch_size, k_prefix, 192)
    text_embeds = torch.randn(batch_size, seq_len, 2560)
    
    loss, similarity = alignment(image_embeds, text_embeds, return_similarity=True)
    
    print(f"Image embeddings: {image_embeds.shape}")
    print(f"Text embeddings: {text_embeds.shape}")
    print(f"Contrastive loss: {loss.item():.4f}")
    print(f"Similarity matrix: {similarity.shape}")
    print(f"Temperature: {alignment.logit_scale.exp().item():.4f}")
    
    # Test cross-modal fusion
    fusion = CrossModalFusionModule(
        image_dim=192,
        text_dim=2560,
        num_heads=8,
        num_layers=2
    )
    
    fused_embeds, attn_weights = fusion(image_embeds, text_embeds, return_attention=True)
    
    print(f"\nCross-modal fusion output: {fused_embeds.shape}")
    print(f"Attention layers: {len(attn_weights)}")
    print(f"Attention shape: {attn_weights[0].shape}")
    
    print("\nImage-text alignment module test passed!")
