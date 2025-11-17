from src.tiny_vlm import TinyVLM
import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


def quantize_bitlinear_weight(weight):
    """
    Quantize weight to 1.58-bit (ternary: {-1, 0, +1})
    """
    scale = weight.abs().mean()

    threshold_low = -0.33 * scale
    threshold_high = 0.33 * scale

    quantized = torch.zeros_like(weight)
    quantized[weight > threshold_high] = 1.0
    quantized[weight < threshold_low] = -1.0

    return quantized, scale


def quantize_bitnet_model(model):
    """
    Quantize BitNet weights to 1.58-bit representation
    """
    print("Quantizing BitNet model to 1.58-bit...")

    quantized_params = {}
    scales = {}

    for name, param in tqdm(model.language_model.named_parameters(), desc="Quantizing weights"):
        if 'weight' in name and param.dim() >= 2:
            quantized, scale = quantize_bitlinear_weight(param.data)
            quantized_params[name] = quantized
            scales[name + '_scale'] = scale
        else:
            quantized_params[name] = param.data

    return quantized_params, scales


def validate_quantization(model_fp16, model_quantized, test_input):
    """
    Validate that quantized model produces similar outputs to fp16 model
    """
    print("Validating quantization...")

    model_fp16.eval()
    model_quantized.eval()

    with torch.no_grad():
        output_fp16 = model_fp16(**test_input)
        output_quantized = model_quantized(**test_input)

    logits_fp16 = output_fp16[0]
    logits_quantized = output_quantized[0]

    mse = torch.nn.functional.mse_loss(logits_fp16, logits_quantized)
    cosine_sim = torch.nn.functional.cosine_similarity(
        logits_fp16.flatten(),
        logits_quantized.flatten(),
        dim=0
    )

    print(f"MSE between fp16 and quantized: {mse.item():.6f}")
    print(f"Cosine similarity: {cosine_sim.item():.6f}")

    return mse.item(), cosine_sim.item()


def calculate_model_size(state_dict):
    """
    Calculate model size in MB
    """
    total_size = 0
    for param in state_dict.values():
        if isinstance(param, torch.Tensor):
            total_size += param.numel() * param.element_size()
    return total_size / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser(
        description='Quantize MicroVLM-M to 1.58-bit BitNet')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fp16 checkpoint')
    parser.add_argument('--output_dir', type=str, default='checkpoints/quantized',
                        help='Output directory for quantized model')
    parser.add_argument('--config', type=str, default='configs/model_config.json',
                        help='Path to model config')
    parser.add_argument('--validate', action='store_true',
                        help='Validate quantization with random inputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    print(f"Loading fp16 model from {args.checkpoint}")
    model = TinyVLM(config_path=args.config, device=args.device)
    model.load_checkpoint(args.checkpoint)
    model.eval()

    fp16_size = calculate_model_size(model.state_dict())
    print(f"FP16 model size: {fp16_size:.2f} MB")

    quantized_params, scales = quantize_bitnet_model(model)

    for name, param in quantized_params.items():
        if name in model.language_model.state_dict():
            model.language_model.state_dict()[name].copy_(param)

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, 'model_quantized.pt')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'quantized_params': quantized_params,
        'scales': scales,
        'config_path': args.config
    }

    torch.save(checkpoint, output_path)
    print(f"Quantized model saved to: {output_path}")

    quantized_size = calculate_model_size(checkpoint['model_state_dict'])
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(
        f"Size reduction: {fp16_size - quantized_size:.2f} MB ({100*(fp16_size - quantized_size)/fp16_size:.1f}%)")

    if args.validate:
        print("\nRunning validation...")
        test_image = torch.randn(1, 3, 224, 224, device=args.device)
        test_tokens = torch.randint(0, 1000, (1, 20), device=args.device)
        test_input = {
            'image': test_image,
            'text_tokens': test_tokens,
            'use_memory': False,
            'return_attention': False
        }

        model_fp16 = TinyVLM(config_path=args.config, device=args.device)
        model_fp16.load_checkpoint(args.checkpoint)

        validate_quantization(model_fp16, model, test_input)

    print("\nQuantization complete!")
    print(f"Output: {output_path}")
    print(f"Final model size: {quantized_size:.2f} MB")


if __name__ == '__main__':
    main()
