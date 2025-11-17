"""
Download Model Weights
Downloads BitNet and DeiT-Tiny pretrained weights
"""

import os
import argparse
import torch
from huggingface_hub import hf_hub_download
import timm

def download_bitnet_weights(output_dir, model_name="1bitLLM/bitnet_b1_58-3B"):
    """
    Download BitNet weights from HuggingFace
    
    Args:
        output_dir: directory to save weights
        model_name: HuggingFace model identifier
    """
    print(f"Downloading BitNet weights: {model_name}")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download model files
        files_to_download = [
            "pytorch_model.bin",
            "config.json",
            "tokenizer.model",
            "tokenizer_config.json"
        ]
        
        for filename in files_to_download:
            print(f"Downloading {filename}...")
            try:
                file_path = hf_hub_download(
                    repo_id=model_name,
                    filename=filename,
                    cache_dir=output_dir
                )
                print(f"  Saved to: {file_path}")
            except Exception as e:
                print(f"  Warning: Could not download {filename}: {e}")
        
        print("\nBitNet weights downloaded successfully!")
        print(f"Location: {output_dir}")
        
    except Exception as e:
        print(f"Error downloading BitNet weights: {e}")
        print("\nManual download instructions:")
        print(f"1. Visit: https://huggingface.co/{model_name}")
        print(f"2. Download model files to: {output_dir}")


def download_deit_weights(output_dir, model_name="deit_tiny_patch16_224"):
    """
    Download DeiT-Tiny weights using timm
    
    Args:
        output_dir: directory to save weights
        model_name: timm model identifier
    """
    print(f"Downloading DeiT-Tiny weights: {model_name}")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create model with pretrained weights
        print("Loading model with pretrained weights...")
        model = timm.create_model(model_name, pretrained=True)
        
        # Save state dict
        save_path = os.path.join(output_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), save_path)
        
        print(f"\nDeiT-Tiny weights saved to: {save_path}")
        print("Download complete!")
        
    except Exception as e:
        print(f"Error downloading DeiT-Tiny weights: {e}")
        print("\nManual download instructions:")
        print("1. Install timm: pip install timm")
        print(f"2. Run: python -c \"import timm; timm.create_model('{model_name}', pretrained=True)\"")


def verify_weights(bitnet_dir, deit_path):
    """
    Verify downloaded weights
    
    Args:
        bitnet_dir: BitNet weights directory
        deit_path: DeiT weights file path
    """
    print("\nVerifying downloaded weights...")
    print("=" * 80)
    
    # Check BitNet
    bitnet_files = os.listdir(bitnet_dir) if os.path.exists(bitnet_dir) else []
    print(f"BitNet directory: {bitnet_dir}")
    print(f"  Files found: {len(bitnet_files)}")
    
    # Check DeiT
    deit_exists = os.path.exists(deit_path)
    print(f"DeiT weights: {deit_path}")
    print(f"  Exists: {deit_exists}")
    
    if deit_exists:
        try:
            checkpoint = torch.load(deit_path, map_location='cpu')
            num_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
            print(f"  Parameters: {num_params:,}")
        except Exception as e:
            print(f"  Error loading: {e}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument('--output_dir', type=str, default='checkpoints/pretrained',
                       help='Output directory for weights')
    parser.add_argument('--bitnet_model', type=str, default='1bitLLM/bitnet_b1_58-3B',
                       help='BitNet model name on HuggingFace')
    parser.add_argument('--deit_model', type=str, default='deit_tiny_patch16_224',
                       help='DeiT model name in timm')
    parser.add_argument('--skip_bitnet', action='store_true', help='Skip BitNet download')
    parser.add_argument('--skip_deit', action='store_true', help='Skip DeiT download')
    
    args = parser.parse_args()
    
    # Create output directories
    bitnet_dir = os.path.join(args.output_dir, 'bitnet')
    deit_dir = os.path.join(args.output_dir, 'deit')
    
    # Download BitNet
    if not args.skip_bitnet:
        download_bitnet_weights(bitnet_dir, args.bitnet_model)
    
    # Download DeiT
    if not args.skip_deit:
        download_deit_weights(deit_dir, args.deit_model)
    
    # Verify
    deit_path = os.path.join(deit_dir, f"{args.deit_model}.pth")
    verify_weights(bitnet_dir, deit_path)
    
    print("\nWeight download complete!")
    print("Next steps:")
    print("  1. Run: python scripts/extract_model_configs.py")
    print("  2. Verify configs in: configs/model_config.json")
    print("  3. Start training with: python scripts/train_stage1.py")


if __name__ == "__main__":
    main()
