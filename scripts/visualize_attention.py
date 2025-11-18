import os
import sys
import json
import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset import CC12MDataset
from attention_visualizer import AttentionVisualizer
from tiny_vlm import TinyVLM


def visualize_cross_attention(model, visualizer, image, text_tokens, output_path, image_path=None):
    model.eval()

    with torch.no_grad():
        logits, memory_state, attn_weights, metadata = model(
            image.unsqueeze(0),
            text_tokens.unsqueeze(0),
            use_memory=True,
            return_attention=True
        )

    if attn_weights is None:
        print("Warning: No attention weights returned")
        return

    cross_attn = visualizer.extract_cross_attention(attn_weights)

    heatmap_path = output_path.replace('.png', '_heatmap.png')
    visualizer.generate_heatmap(cross_attn, heatmap_path)

    if image_path and os.path.exists(image_path):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        img_pil = Image.open(image_path).resize((224, 224))
        axes[0].imshow(img_pil)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        attn_map = cross_attn.mean(dim=1).cpu().numpy()
        axes[1].imshow(attn_map, cmap='viridis', aspect='auto')
        axes[1].set_title('Cross-Attention (Avg over heads)')
        axes[1].set_xlabel('Text Tokens')
        axes[1].set_ylabel('Image Patches')

        overlay_attn = attn_map.mean(axis=1).reshape(14, 14)
        overlay_attn = (overlay_attn - overlay_attn.min()) / \
            (overlay_attn.max() - overlay_attn.min() + 1e-8)
        overlay_attn = np.array(Image.fromarray(
            (overlay_attn * 255).astype(np.uint8)).resize((224, 224)))

        axes[2].imshow(img_pil)
        axes[2].imshow(overlay_attn, cmap='jet', alpha=0.5)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization: {output_path}")
    else:
        print(f"Saved heatmap: {heatmap_path}")

    return cross_attn


def visualize_memory_addressing(model, visualizer, image, text_tokens, output_path):
    model.eval()

    with torch.no_grad():
        logits, memory_state, attn_weights, metadata = model(
            image.unsqueeze(0),
            text_tokens.unsqueeze(0),
            use_memory=True,
            return_attention=True
        )

    if memory_state is None or 'addressing_weights' not in memory_state:
        print("Warning: No memory addressing weights available")
        return

    addressing_weights = memory_state['addressing_weights'].cpu().numpy()
    memory_vector = memory_state.get('memory_vector', None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(addressing_weights[0])), addressing_weights[0])
    axes[0].set_title('Memory Addressing Weights')
    axes[0].set_xlabel('Memory Slot')
    axes[0].set_ylabel('Weight')
    axes[0].grid(True, alpha=0.3)

    if memory_vector is not None:
        memory_vis = memory_vector.cpu().numpy().reshape(1, -1)
        im = axes[1].imshow(memory_vis, cmap='viridis', aspect='auto')
        axes[1].set_title('Memory Vector')
        axes[1].set_xlabel('Dimension')
        axes[1].set_yticks([])
        plt.colorbar(im, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, 'No memory vector', ha='center', va='center')
        axes[1].set_title('Memory Vector (N/A)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved memory visualization: {output_path}")


def visualize_attention_divergence(visualizer, attn_weights, output_path):
    if attn_weights is None:
        print("Warning: No attention weights for divergence analysis")
        return

    cross_attn = visualizer.extract_cross_attention(attn_weights)

    divergence_scores = []
    num_heads = cross_attn.size(1)

    for head_idx in range(num_heads):
        attn_head = cross_attn[:, head_idx, :, :]
        score = visualizer.fast_epps_pulley(
            attn_head.flatten(), attn_head.flatten())
        divergence_scores.append(score.item())

    plt.figure(figsize=(10, 6))
    plt.bar(range(num_heads), divergence_scores)
    plt.title('Attention Divergence by Head (EPPs-Pulley Test)')
    plt.xlabel('Attention Head')
    plt.ylabel('Divergence Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved divergence analysis: {output_path}")

    return divergence_scores


def batch_visualize(model, visualizer, dataset, num_samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    indices = np.random.choice(len(dataset), min(
        num_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="Generating visualizations"):
        sample = dataset[idx]
        if sample is None:
            continue

        image, text_tokens = sample
        image_path = dataset.metadata['samples'][idx]['image_path']

        output_prefix = os.path.join(output_dir, f'sample_{idx}')

        try:
            visualize_cross_attention(
                model, visualizer, image, text_tokens,
                f'{output_prefix}_cross_attention.png',
                image_path
            )
        except Exception as e:
            print(f"Error visualizing cross-attention for sample {idx}: {e}")

        try:
            visualize_memory_addressing(
                model, visualizer, image, text_tokens,
                f'{output_prefix}_memory.png'
            )
        except Exception as e:
            print(f"Error visualizing memory for sample {idx}: {e}")

    print(f"\nBatch visualization complete. Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize attention patterns for MicroVLM-M')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/model_config.json',
                        help='Path to model config')
    parser.add_argument('--dataset', type=str, default='data/cc12m/val_metadata.json',
                        help='Path to dataset metadata')
    parser.add_argument('--image_dir', type=str, default='data/cc12m/images',
                        help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='logs/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    model = TinyVLM(config_path=args.config, device=args.device)
    model.load_checkpoint(args.checkpoint)
    model.eval()

    with open(args.config, 'r') as f:
        config = json.load(f)
    visualizer = AttentionVisualizer(config['attention_visualizer'])

    print(f"Loading dataset from {args.dataset}")
    dataset = CC12MDataset(
        metadata_path=args.dataset,
        image_dir=args.image_dir,
        image_size=224
    )

    batch_visualize(model, visualizer, dataset,
                    args.num_samples, args.output_dir)

    print("\nVisualization complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
