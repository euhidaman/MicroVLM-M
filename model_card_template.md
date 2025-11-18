---
license: apache-2.0
tags:
- vision
- language
- multimodal
- bitnet
- larimar
- episodic-memory
- evo-1
- deit
- vision-language-model
datasets:
- cc12m
language:
- en
metrics:
- perplexity
- contrastive-loss
library_name: pytorch
---

# MicroVLM-M: Tiny Vision-Language Model

**MicroVLM-M** is a compact, efficient vision-language model that combines state-of-the-art architectural components to achieve strong multimodal understanding while maintaining a small footprint (<500MB).

## Model Architecture

### Core Components

1. **Vision Encoder: DeiT-Tiny**
   - Image size: 224×224
   - Patch size: 16×16 (196 patches)
   - Embedding dimension: 192
   - 12 transformer layers, 3 attention heads
   - Pre-trained on ImageNet-1k

2. **Language Model: BitNet-3B (1.58-bit quantized)**
   - Hidden size: 2560
   - 30 transformer layers
   - Grouped Query Attention (GQA): 20 query heads, 5 KV heads
   - Head dimension: 128
   - Vocabulary: 128,256 tokens
   - Max sequence length: 4096
   - RoPE positional embeddings (θ=500k)
   - Ultra-efficient 1.58-bit quantization

3. **Multimodal Adapter**
   - Projects vision embeddings (192D) → language space (2560D)
   - Learnable pooling: 196 patches → 25 prefix tokens
   - MLP with 512 hidden units, 0.1 dropout
   - Bridges vision and language modalities

4. **Larimar Episodic Memory**
   - 128 memory slots × 2560 dimensions
   - Generalized Predictive Memory (GPM) mechanism
   - Direct writing via pseudoinverse approximation
   - KV cache injection across all 30 LM layers
   - Memory-augmented context for enhanced recall

5. **ScopeNet Decision Module**
   - Dynamic memory gating (decides when to apply memory)
   - 2-layer MLP (2560 → 256 → 1)
   - Adaptive context-aware memory retrieval

6. **EVO-1 Image-Text Alignment**
   - Contrastive learning module (projection_dim=512, temperature=0.07)
   - Cross-modal fusion with 8-head attention, 2 layers
   - Explicit image-text representation alignment

### Model Size & Efficiency

- **Total Parameters**: ~3.2B (BitNet 1.58-bit quantization drastically reduces memory)
- **Actual Model Size**: <500 MB
- **Trainable Parameters (Stage 1)**: ~15-20% (adapters, memory, alignment modules)
- **Frozen Components**: Early DeiT layers (8/12), most BitNet layers (26/30)

## Training

### Stage 1: Adapter & Memory Training
- **Dataset**: CC12M (Conceptual Captions 12M)
  - 110,002 training images
  - 5,789 validation images
- **Batch Size**: 10 (effective batch: 20 with 2× gradient accumulation)
- **Learning Rate**: 1e-4 (cosine decay to 1e-6)
- **Optimizer**: AdamW (β₁=0.9, β₂=0.999, weight_decay=0.01)
- **Mixed Precision**: FP16 AMP enabled
- **Max Steps**: 50,000
- **Loss Components**:
  - Language Modeling Loss (weight: 1.0)
  - EVO-1 Alignment Loss (weight: 0.5)
  - Memory Reconstruction Loss (weight: 0.1)
  - Scope Decision Loss (weight: 0.01)

### Stage 2: Fine-tuning (Planned)
- Unfreeze last 4 BitNet layers
- Lower learning rate (5e-5)
- Task-specific fine-tuning

## Usage

### Installation

```bash
pip install torch torchvision transformers timm pillow huggingface-hub
```

### Inference Example

```python
import torch
from PIL import Image
from tiny_vlm import TinyVLM

# Load model
model = TinyVLM(config_path="configs/model_config.json", device="cuda")
model.eval()

# Load image
image = Image.open("example.jpg").convert("RGB")

# Simple character-level tokenization for caption
caption = "a photo of a cat"
token_ids = torch.tensor([ord(c) - ord('a') + 1 if c.isalpha() else 27 for c in caption.lower()])
token_ids = token_ids.unsqueeze(0).to("cuda")  # (1, text_len)

# Forward pass
with torch.no_grad():
    logits, memory_state, attn_weights, metadata = model(
        images=[image],
        text_token_ids=token_ids,
        use_memory=True,
        use_fusion=False,
        return_attention=False
    )

# logits: (1, seq_len, vocab_size)
print(f"Output logits shape: {logits.shape}")
```

### Training from Checkpoint

```python
from train_stage1 import Stage1Trainer
import json

# Load config
with open("configs/stage1_config.json") as f:
    config = json.load(f)

# Initialize trainer
trainer = Stage1Trainer(config)

# Resume training
trainer.train()
```

## Technical Highlights

- **Memory Efficiency**: BitNet 1.58-bit quantization reduces memory footprint by ~10×
- **Grouped Query Attention**: 4:1 ratio (20 Q heads : 5 KV heads) reduces compute/memory
- **Episodic Memory**: Larimar GPM enables long-term context retention beyond transformer window
- **Dynamic Memory Gating**: ScopeNet adaptively decides when memory improves predictions
- **EVO-1 Alignment**: Explicit contrastive learning ensures vision-language feature alignment

## Evaluation Metrics

- **Perplexity**: Measures language modeling performance
- **Alignment Loss**: Contrastive image-text similarity
- **Memory KL Divergence**: Regularizes memory update efficiency
- **Scope Decision Accuracy**: Measures memory gating effectiveness

## Limitations

- Character-level tokenization (simple baseline; BPE/SentencePiece recommended for production)
- Trained on English CC12M captions only
- Vision encoder frozen during Stage 1 (limited visual feature adaptation)
- Requires GPU with ≥20GB VRAM for training (batch_size=10)

## Citation

```bibtex
@misc{microvlm-m-2025,
  title={MicroVLM-M: A Compact Vision-Language Model with Episodic Memory},
  author={Your Name},
  year={2025},
  howpublished={\url{https://huggingface.co/euhidaman/MicroVLM-M}}
}
```

## License

Apache 2.0

## Acknowledgements

- **BitNet**: Microsoft Research (1.58-bit quantization)
- **DeiT**: Meta AI (Data-efficient Image Transformers)
- **Larimar**: Generalized Predictive Memory framework
- **EVO**: Image-text alignment methodology
- **CC12M**: Google Conceptual Captions dataset

---

**Model Status**: Stage 1 Training in Progress  
**Last Updated**: {timestamp}  
**Training Step**: {global_step}  
**Validation Loss**: {val_loss:.4f}
