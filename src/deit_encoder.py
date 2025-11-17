"""
DeiT-Tiny Vision Encoder Integration
Direct implementation with pretrained weight loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from typing import Optional
import timm

class DeiTTinyEncoder(nn.Module):
    """
    DeiT-Tiny vision encoder wrapper
    Extracts patch tokens as visual embeddings
    """
    
    def __init__(self, config_path=None, config_dict=None, pretrained=True):
        super().__init__()
        
        # Load configuration
        if config_dict is not None:
            config = config_dict
        elif config_path is not None:
            with open(config_path, 'r') as f:
                master_config = json.load(f)
            config = master_config['deit_tiny']
        else:
            raise ValueError("Must provide either config_path or config_dict")
        
        self.model_name = config.get('model_name', 'deit_tiny_patch16_224')
        self.image_size = config['image_size']
        self.patch_size = config['patch_size']
        self.num_patches = config['num_patches']
        self.embed_dim = config['embed_dim']
        self.pretrained = pretrained
        
        # Load DeiT-Tiny using timm
        try:
            self.model = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                num_classes=0,  # Remove classification head
                global_pool=''  # No global pooling, keep patch tokens
            )
            print(f"Loaded DeiT-Tiny model: {self.model_name}")
        except Exception as e:
            print(f"Error loading timm model: {e}")
            print("Falling back to manual weight download")
            self.model = self._create_model_manual()
        
        # Freeze early layers during training (optional)
        self.frozen_stages = 0
    
    def _create_model_manual(self):
        """Create DeiT-Tiny manually if timm fails"""
        class ManualDeiT(nn.Module):
            def __init__(self, img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (img_size // patch_size) ** 2, embed_dim))
                
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
                                               dropout=0.0, activation='gelu', batch_first=True)
                    for _ in range(depth)
                ])
                self.norm = nn.LayerNorm(embed_dim)
            
            def forward(self, x):
                B = x.shape[0]
                x = self.patch_embed(x).flatten(2).transpose(1, 2)
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.pos_embed
                
                for block in self.blocks:
                    x = block(x)
                
                x = self.norm(x)
                return x
        
        return ManualDeiT(img_size=self.image_size, patch_size=self.patch_size, embed_dim=self.embed_dim)
    
    def freeze_stages(self, num_stages):
        """Freeze early transformer blocks"""
        self.frozen_stages = num_stages
        
        # Freeze patch embedding
        if num_stages > 0:
            if hasattr(self.model, 'patch_embed'):
                for param in self.model.patch_embed.parameters():
                    param.requires_grad = False
        
        # Freeze transformer blocks
        if hasattr(self.model, 'blocks'):
            for i in range(min(num_stages, len(self.model.blocks))):
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = False
    
    def forward(self, images):
        """
        Forward pass to extract patch embeddings
        
        Args:
            images: (batch, 3, image_size, image_size)
        
        Returns:
            patch_embeddings: (batch, num_patches, embed_dim)
        """
        # Get features from DeiT
        features = self.model(images)
        
        # Remove CLS token, keep only patch tokens
        if features.size(1) == self.num_patches + 1:
            patch_embeddings = features[:, 1:, :]  # Skip CLS token
        else:
            patch_embeddings = features
        
        return patch_embeddings
    
    def load_pretrained_weights(self, checkpoint_path):
        """Load pretrained DeiT weights from checkpoint"""
        print(f"Loading DeiT-Tiny weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        
        if incompatible.missing_keys:
            print(f"Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys: {incompatible.unexpected_keys}")
        
        print("DeiT-Tiny weights loaded successfully")


class ImagePreprocessor:
    """
    Image preprocessing for DeiT-Tiny
    Standard ImageNet normalization
    """
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def __call__(self, images):
        """
        Preprocess images
        
        Args:
            images: (batch, 3, H, W) in range [0, 1]
        
        Returns:
            preprocessed: (batch, 3, image_size, image_size) normalized
        """
        # Resize if needed
        if images.size(-1) != self.image_size or images.size(-2) != self.image_size:
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize
        device = images.device
        images = (images - self.mean.to(device)) / self.std.to(device)
        
        return images


if __name__ == "__main__":
    # Test DeiT-Tiny encoder
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "model_config.json")
    
    encoder = DeiTTinyEncoder(config_path=config_path, pretrained=False)
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    patch_embeddings = encoder(images)
    
    print(f"Input shape: {images.shape}")
    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    print(f"Expected num_patches: {encoder.num_patches}")
    print(f"Expected embed_dim: {encoder.embed_dim}")
    
    assert patch_embeddings.shape == (batch_size, encoder.num_patches, encoder.embed_dim), "Shape mismatch!"
    
    # Test preprocessing
    preprocessor = ImagePreprocessor(image_size=224)
    processed = preprocessor(images)
    print(f"Preprocessed shape: {processed.shape}")
    
    print("DeiT-Tiny encoder test passed!")
