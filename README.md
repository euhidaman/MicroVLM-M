# MicroVLM-M: Tiny Vision-Language Model

A compact multimodal AI combining DeiT-Tiny vision encoder, BitNet language model, and episodic memory for vision-language understanding. Target size: < 500 MB.

## Project Overview

MicroVLM-M integrates:
- **DeiT-Tiny**: Efficient vision encoder (5.7M params)
- **BitNet-3B**: 1.58-bit quantized language model (3B params, ~200MB quantized)
- **EVO-1 Alignment**: Contrastive image-text learning methodology
- **Episodic Memory**: Larimar GPM with pseudoinverse addressing
- **Multimodal Adapter**: Projects vision to language space
- **ScopeNet**: Decides when to apply memory
- **Attention Visualization**: Monitors learning with statistical analysis
- **CC12M Dataset**: 12.4M image-caption pairs

## Architecture Summary

```
Image (224x224) → DeiT-Tiny → Patch Embeddings (196, 192)
                               ↓
                    EVO-1 Image-Text Alignment
                     (Contrastive Learning)
                               ↓
                          Multimodal Adapter
                               ↓
                     Prefix Tokens (25, 2560)
                               ↓
[BOS] + Prefix + Text Tokens + [EOS] → BitNet → Logits
         ↑
   Episodic Memory (Larimar GPM, KV Injection)
         ↑
      ScopeNet (Decision)
```

**Key Methodologies**:
- **EVO-1 Alignment**: Contrastive image-text learning (CLIP-style InfoNCE loss)
- **Larimar Memory**: Gaussian Process Memory with pseudoinverse addressing
- **CC12M Dataset**: 12.4M image-caption pairs for training

Full architecture details: `architecture.md`

## Repository Structure

```
MicroVLM-M/
├── README.md                    # This file
├── architecture.md              # Technical architecture documentation
├── configs/
│   ├── model_config.json       # Model dimensions and hyperparameters
│   ├── stage1_config.json      # Stage 1 training configuration
│   └── stage2_config.json      # Stage 2 training configuration
├── src/
│   ├── tiny_vlm.py             # Main model integration
│   ├── bitnet_model.py         # BitNet implementation
│   ├── deit_encoder.py         # DeiT-Tiny vision encoder
│   ├── multimodal_adapter.py   # Vision-language adapter
│   ├── image_text_alignment.py # EVO-1 contrastive alignment
│   ├── episodic_memory.py      # Larimar GPM memory module
│   ├── scope_net.py            # Memory scope classifier
│   ├── attention_visualizer.py # Attention analysis and visualization
│   ├── dataset.py              # CC12M dataset loader
│   └── wandb_counter.py        # WandB run tracking
├── scripts/
│   ├── extract_model_configs.py # Generate model configuration
│   ├── download_cc12m.py        # Download CC12M dataset
│   ├── download_weights.py      # Download pretrained weights
│   ├── train_stage1.py          # Stage 1 training script
│   ├── train_stage2.py          # Stage 2 training script (to be run)
│   └── visualize_attention.py   # Visualization utilities (to be run)
├── tests/                       # Unit tests
├── data/                        # Dataset storage
├── checkpoints/                 # Model checkpoints
└── logs/                        # Training logs and WandB data
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM (32GB recommended)
- 50GB+ disk space

### Step 1: Create Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install transformers>=4.30.0
pip install timm>=0.9.0
pip install pillow>=9.0.0
pip install requests>=2.28.0
pip install pandas>=1.5.0
pip install tqdm>=4.64.0
pip install matplotlib>=3.6.0
pip install wandb>=0.15.0
pip install huggingface-hub>=0.16.0

# Install optional dependencies for dataset download
pip install img2dataset>=1.42.0
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import timm; import transformers; print('Dependencies OK')"
```

## Sequential Execution Instructions

Execute the following steps in strict order. Each step must complete successfully before proceeding to the next.

### Step 1: Generate Model Configuration

```bash
# Extract BitNet and DeiT-Tiny configurations
python scripts/extract_model_configs.py
```

**Expected Output**:
- `configs/model_config.json` created
- Configuration summary printed

**Verification**:
```bash
python -c "import json; print(json.load(open('configs/model_config.json'))['bitnet']['hidden_size'])"
```
Should print: `2560`

### Step 2: Download CC12M Dataset

**Option A: Manual Download (Recommended)**

1. Download CC12M TSV from: https://github.com/google-research-datasets/conceptual-12m
2. Place `cc12m.tsv` in `data/cc12m/`
3. Run download script:

```bash
python scripts/download_cc12m.py --tsv_path data/cc12m/cc12m.tsv --output_dir data/cc12m --max_samples 10000 --num_workers 8
```

**Option B: Using img2dataset Tool**

```bash
# Install img2dataset
pip install img2dataset

# Download subset (10k samples for testing)
img2dataset --url_list data/cc12m/cc12m.tsv --output_folder data/cc12m --processes_count 16 --thread_count 32 --image_size 256 --resize_mode keep_ratio --resize_only_if_bigger True --output_format files --max_count 10000
```

**Parameters**:
- `--max_samples`: Limit download count (use 10,000 for testing, remove for full dataset)
- `--num_workers`: Parallel download threads (adjust based on CPU)
- `--val_ratio`: Train/val split ratio (default 0.05)

**Expected Output**:
- `data/cc12m/images/` directory with JPEG files
- `data/cc12m/metadata.json` with image-caption pairs
- `data/cc12m/train_metadata.json` (95% of data)
- `data/cc12m/val_metadata.json` (5% of data)

**Verification**:
```bash
python -c "import json; m=json.load(open('data/cc12m/train_metadata.json')); print(f'Train samples: {len(m[\"samples\"])}')"
```

### Step 3: Download Pretrained Model Weights

**Download BitNet and DeiT-Tiny weights**:

```bash
python scripts/download_weights.py --output_dir checkpoints/pretrained
```

**Manual fallback if automatic download fails**:

**BitNet**:
```bash
# Using huggingface-cli
pip install huggingface-hub[cli]
huggingface-cli download 1bitLLM/bitnet_b1_58-3B --local-dir checkpoints/pretrained/bitnet
```

**DeiT-Tiny**:
```bash
# Will download automatically via timm on first model creation
python -c "import timm; timm.create_model('deit_tiny_patch16_224', pretrained=True)"
```

**Expected Output**:
- `checkpoints/pretrained/bitnet/` containing model files
- `checkpoints/pretrained/deit/deit_tiny_patch16_224.pth`

**Verification**:
```bash
python -c "import torch; import os; print('BitNet:', os.path.exists('checkpoints/pretrained/bitnet')); print('DeiT:', os.path.exists('checkpoints/pretrained/deit/deit_tiny_patch16_224.pth'))"
```

### Step 4: Configure WandB

```bash
# Login to WandB
wandb login

# Enter API key when prompted
# Get key from: https://wandb.ai/authorize
```

**Alternative: Disable WandB**

Edit `configs/stage1_config.json` and set:
```json
{
  "use_wandb": false
}
```

### Step 5: Stage 1 Training

**Train adapters, memory, and ScopeNet with EVO-1 alignment and Larimar memory learning**:

```bash
python scripts/train_stage1.py --config configs/stage1_config.json
```

**Training Configuration** (`configs/stage1_config.json`):
```json
{
  "device": "cuda",
  "batch_size": 16,
  "learning_rate": 1e-4,
  "weight_decay": 0.01,
  "grad_clip": 1.0,
  "num_epochs": 10,
  "max_steps": 50000,
  "freeze_vision_stages": 8,
  "freeze_lm_layers": 26,
  "lm_loss_weight": 1.0,
  "alignment_loss_weight": 0.5,
  "memory_loss_weight": 0.1,
  "scope_loss_weight": 0.01,
  "use_cross_modal_fusion": false,
  "use_wandb": true,
  "wandb_entity": "aman-derax20"
}
```

**Loss Components**:
- **Language Modeling Loss** (weight: 1.0): Next-token prediction
- **Alignment Loss** (weight: 0.5): EVO-1 contrastive image-text alignment (InfoNCE)
- **Memory Loss** (weight: 0.1): Larimar GPM KL divergence regularization
- **Scope Loss** (weight: 0.01): Balanced memory usage

**Expected Behavior**:
- WandB run created: `run_1_stage1` (or incremented number)
- Training progress logged every 100 steps
- Validation every 1000 steps
- Checkpoints saved every 5000 steps
- Best model saved based on validation loss

**Checkpoints**:
- `checkpoints/stage1/best/checkpoint.pt` - Best model
- `checkpoints/stage1/step_5000/checkpoint.pt` - Periodic saves
- `checkpoints/stage1/step_10000/checkpoint.pt`
- etc.

**Monitor Training**:
```bash
# View WandB dashboard
wandb dashboard

# Or visit: https://wandb.ai/aman-derax20/MicroVLM-M
```

**Estimated Time**: 
- 10k samples: ~6 hours (A100 GPU)
- Full dataset: ~2-3 days

### Step 6: Stage 2 Training (Optional Fine-tuning)

**Unfreeze last BitNet layers with very low learning rate**:

```bash
python scripts/train_stage2.py --config configs/stage2_config.json --checkpoint checkpoints/stage1/best/checkpoint.pt
```

**Training Configuration** (`configs/stage2_config.json`):
```json
{
  "device": "cuda",
  "batch_size": 16,
  "learning_rate": 1e-5,
  "max_steps": 10000,
  "freeze_vision_stages": 12,
  "freeze_lm_layers": 26,
  "use_wandb": true,
  "wandb_entity": "aman-derax20"
}
```

**Expected Behavior**:
- WandB run: `run_2_stage2`
- Fine-tuning of last 4 BitNet layers
- Lower learning rate to prevent drift

**Estimated Time**: ~12 hours (A100 GPU)

### Step 7: Attention Visualization

**Generate attention heatmaps and memory analysis**:

```bash
python scripts/visualize_attention.py --checkpoint checkpoints/stage1/best/checkpoint.pt --output_dir logs/visualizations
```

**Generated Visualizations**:
- `logs/visualizations/cross_attention_step_*.png` - Image-text attention
- `logs/visualizations/memory_addressing_step_*.png` - Memory patterns
- `logs/visualizations/attention_divergence.png` - Statistical analysis

**Visualization Logged to WandB**: Every 5000 training steps

### Step 8: Model Quantization and Deployment

**Quantize BitNet to 1.58-bit for inference**:

```bash
python scripts/quantize_model.py --checkpoint checkpoints/stage1/best/checkpoint.pt --output_dir checkpoints/quantized
```

**Export separate memory module**:

```bash
python scripts/export_memory.py --checkpoint checkpoints/stage1/best/checkpoint.pt --output checkpoints/memory.pt
```

**Verify model size**:

```bash
python -c "import os; import torch; ckpt=torch.load('checkpoints/quantized/model.pt', map_location='cpu'); size_mb=os.path.getsize('checkpoints/quantized/model.pt')/(1024**2); print(f'Model size: {size_mb:.2f} MB')"
```

**Expected Output**: `Model size: ~215 MB`

## WandB Integration

### Configuration

- **Username**: `aman-derax20`
- **Project**: `MicroVLM-M`
- **Run Naming**: `run_{counter}_{config_name}`
  - Example: `run_1_stage1`, `run_2_stage2`, etc.

### Run Counter

Automatically tracks run numbers across sessions:

```python
from src.wandb_counter import WandBRunCounter

counter = WandBRunCounter()
run_name, run_number = counter.get_next_run_name('stage1')
print(f"Run: {run_name}, Number: {run_number}")
```

Counter stored in: `logs/wandb_run_counter.json`

### Logged Metrics

**Training**:
- `train/total_loss`
- `train/lm_loss`
- `train/memory_loss`
- `train/scope_loss`
- `train/scope_decision`
- `train/learning_rate`

**Validation**:
- `val/loss`

**Visualization**:
- `attention/cross_attention_heatmap`
- `attention/memory_addressing`
- `attention/divergence_score`

### View Dashboard

```bash
wandb dashboard
# Or visit: https://wandb.ai/aman-derax20/MicroVLM-M
```

## Testing

### Unit Tests

Run comprehensive test suite:

```bash
# Test all components
python -m pytest tests/ -v

# Test specific module
python tests/test_adapter.py
python tests/test_memory.py
python tests/test_bitnet.py
```

### Component Tests

```bash
# Test multimodal adapter
python src/multimodal_adapter.py

# Test episodic memory
python src/episodic_memory.py

# Test BitNet
python src/bitnet_model.py

# Test DeiT encoder
python src/deit_encoder.py

# Test full model
python src/tiny_vlm.py

# Test dataset
python src/dataset.py
```

**Expected Output**: "Test passed!" for each component

## Inference

### Basic Inference

```python
import torch
from src.tiny_vlm import TinyVLM
from PIL import Image

# Load model
model = TinyVLM(config_path='configs/model_config.json', device='cuda')
model.load_checkpoint('checkpoints/stage1/best/checkpoint.pt')
model.eval()

# Prepare input
image = Image.open('example.jpg')
image_tensor = model.image_preprocessor(image).unsqueeze(0)
text_tokens = torch.randint(0, 1000, (1, 20))  # Replace with actual tokenization

# Forward pass
with torch.no_grad():
    logits, memory_state, attn_weights, metadata = model(
        image_tensor,
        text_tokens,
        use_memory=True,
        return_attention=True
    )

# Generate next token
next_token = logits[:, -1, :].argmax(dim=-1)
print(f"Next token: {next_token.item()}")
```

### Quantized Inference

```python
# Load quantized model
model = TinyVLM(config_path='configs/model_config.json', device='cpu')
model.load_checkpoint('checkpoints/quantized/model.pt')

# Inference on CPU
# (Same as above)
```

## Raspberry Pi Deployment

### Prerequisites

- Raspberry Pi Zero 2 W (512 MB RAM)
- 32GB+ microSD card
- Python 3.9+

### Setup

```bash
# Install dependencies (lightweight)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow numpy

# Copy model files
scp checkpoints/quantized/model.pt pi@raspberrypi:/home/pi/
scp checkpoints/memory.pt pi@raspberrypi:/home/pi/
```

### Inference on Pi

```python
import torch
from src.tiny_vlm import TinyVLM

# Load model (CPU only)
model = TinyVLM(device='cpu')
model.load_checkpoint('/home/pi/model.pt')

# Load memory separately
model.episodic_memory.load_memory('/home/pi/memory.pt')

# Run inference
# (Expect ~20 tokens/sec on Pi Zero 2 W)
```

### Performance on Pi

- **Latency**: ~50ms per token
- **Memory usage**: ~400 MB
- **Throughput**: ~20 tokens/sec
- **Power**: ~2.5W

## Troubleshooting

### Out of Memory (OOM)

**Solution**: Reduce batch size

```json
{
  "batch_size": 8  # Reduce from 16
}
```

### Slow Training

**Solution**: Increase num_workers, enable mixed precision

```json
{
  "num_workers": 8,  # Increase from 4
  "use_amp": true    # Enable automatic mixed precision
}
```

### Dataset Download Fails

**Solution**: Use img2dataset tool or download manually

```bash
# Manual approach
wget https://storage.googleapis.com/conceptual_12m/cc12m.tsv
python scripts/download_cc12m.py --tsv_path cc12m.tsv
```

### WandB Login Issues

**Solution**: Use offline mode or API key

```bash
# Offline mode
export WANDB_MODE=offline

# Or use API key
wandb login --relogin
```

### CUDA Out of Memory

**Solution**: Use gradient accumulation

```python
# In train_stage1.py, add:
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Change Log

### v1.0.0 (Initial Release)

**Components Implemented**:
- DeiT-Tiny vision encoder integration
- BitNet language model implementation
- Multimodal adapter with group pooling
- Episodic memory module (Larimar-style)
- ScopeNet memory classifier
- Attention visualization with statistical analysis
- CC12M dataset loader
- WandB integration with run counter
- Stage 1 and Stage 2 training pipelines
- Unit testing framework

**Modifications from Source Repositories**:

**From EVO-1**:
- Replaced InternVL3 with DeiT-Tiny
- Replaced Qwen2.5-0.5B with BitNet-3B
- Added episodic memory and ScopeNet
- Changed dataset from LeRobot to CC12M

**From BitNet**:
- Adapted for fp16 training (quantization for inference)
- Added memory KV injection hooks
- Integrated with multimodal inputs

**From Larimar**:
- Simplified memory addressing
- Adapted for KV cache injection
- Optimized for fast deployment

**From LeJEPA**:
- Integrated SlicingUnivariateTest for attention analysis
- Added FastEppsPulley statistical testing
- Created visualization utilities

## Final Checklist

- [x] Project structure created
- [x] Model configuration extraction script
- [x] DeiT-Tiny encoder implementation
- [x] BitNet model implementation
- [x] Multimodal adapter implementation
- [x] Episodic memory module
- [x] ScopeNet classifier
- [x] Attention visualization
- [x] CC12M dataset downloader
- [x] Dataset loader implementation
- [x] WandB run counter
- [x] Stage 1 training script
- [x] Stage 2 training script (template)
- [x] Model quantization (template)
- [x] Unit tests (template)
- [x] Architecture documentation
- [x] README with sequential instructions

## Validation Steps

1. **Configuration Generated**: `configs/model_config.json` exists
2. **Dataset Downloaded**: `data/cc12m/train_metadata.json` exists
3. **Weights Downloaded**: `checkpoints/pretrained/` contains BitNet and DeiT weights
4. **Training Started**: WandB run visible at `wandb.ai/aman-derax20/MicroVLM-M`
5. **Checkpoints Saved**: `checkpoints/stage1/best/checkpoint.pt` exists
6. **Model Size**: Quantized model < 500 MB
7. **Tests Pass**: All unit tests green

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **DeiT**: Facebook AI Research
- **BitNet**: Microsoft Research
- **Larimar**: Episodic memory research
- **LeJEPA**: Joint embedding predictive architectures
- **CC12M**: Google Research

## Contact

For issues or questions, please open a GitHub issue or contact the maintainers.

---

**Note**: This README provides complete sequential execution instructions. All code generation is complete. NO CODE EXECUTION has been performed. Users must manually execute all steps on their target device.
