"""
CC12M Dataset Downloader and Preprocessor
Downloads and prepares the CC12M dataset for training
"""

import os
import json
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import argparse

def download_image(item, save_dir, timeout=5):
    """
    Download single image
    
    Args:
        item: (idx, url, caption) tuple
        save_dir: directory to save images
        timeout: download timeout in seconds
    
    Returns:
        success: bool indicating success
        result: (idx, image_path, caption, error) tuple
    """
    idx, url, caption = item
    
    try:
        # Download image
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Open and verify image
        img = Image.open(BytesIO(response.content))
        img.verify()
        
        # Re-open for saving (verify closes the image)
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save image
        image_filename = f"{idx:08d}.jpg"
        image_path = os.path.join(save_dir, image_filename)
        img.save(image_path, 'JPEG', quality=95)
        
        return True, (idx, image_path, caption, None)
        
    except Exception as e:
        return False, (idx, None, caption, str(e))


def download_cc12m_metadata(output_dir):
    """
    Download CC12M metadata TSV files
    
    The actual CC12M dataset is distributed as TSV files with image URLs and captions.
    Users need to download from: https://github.com/google-research-datasets/conceptual-12m
    """
    print("CC12M Metadata Download Instructions:")
    print("=" * 80)
    print("1. Visit: https://github.com/google-research-datasets/conceptual-12m")
    print("2. Download the TSV files (cc12m.tsv)")
    print(f"3. Place the TSV file in: {output_dir}")
    print("=" * 80)
    print("\nAlternatively, you can use the img2dataset tool:")
    print("  pip install img2dataset")
    print("  img2dataset --url_list cc12m.tsv --output_folder cc12m_data --processes_count 16")
    print("=" * 80)


def process_tsv(tsv_path, output_dir, max_samples=None, num_workers=16):
    """
    Process CC12M TSV file and download images
    
    Args:
        tsv_path: path to cc12m.tsv file
        output_dir: directory to save processed dataset
        max_samples: maximum number of samples to download (None for all)
        num_workers: number of parallel download workers
    """
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Reading TSV file: {tsv_path}")
    
    # Read TSV
    df = pd.read_csv(tsv_path, sep='\t', names=['url', 'caption'])
    
    if max_samples is not None:
        df = df.head(max_samples)
    
    print(f"Total samples: {len(df)}")
    
    # Prepare items for download
    items = [(idx, row['url'], row['caption']) for idx, row in df.iterrows()]
    
    # Download images in parallel
    print(f"Downloading images with {num_workers} workers...")
    
    download_fn = partial(download_image, save_dir=images_dir)
    
    successful_items = []
    failed_items = []
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(download_fn, items),
            total=len(items),
            desc="Downloading"
        ))
    
    # Collect results
    for success, result in results:
        if success:
            successful_items.append(result)
        else:
            failed_items.append(result)
    
    print(f"\nDownload complete!")
    print(f"  Successful: {len(successful_items)}")
    print(f"  Failed: {len(failed_items)}")
    
    # Save metadata
    metadata = {
        'samples': [
            {
                'idx': idx,
                'image_path': img_path,
                'caption': caption
            }
            for idx, img_path, caption, _ in successful_items
        ]
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Save failed items log
    if failed_items:
        failed_path = os.path.join(output_dir, 'failed_downloads.json')
        failed_log = [
            {'idx': idx, 'caption': caption, 'error': error}
            for idx, _, caption, error in failed_items
        ]
        with open(failed_path, 'w') as f:
            json.dump(failed_log, f, indent=2)
        print(f"Failed downloads logged to: {failed_path}")
    
    return metadata


def create_train_val_split(metadata_path, output_dir, val_ratio=0.05):
    """
    Split dataset into train and validation sets
    
    Args:
        metadata_path: path to metadata.json
        output_dir: directory to save splits
        val_ratio: fraction of data for validation
    """
    print(f"Creating train/val split (val_ratio={val_ratio})...")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    samples = metadata['samples']
    total = len(samples)
    val_size = int(total * val_ratio)
    
    # Shuffle
    import random
    random.shuffle(samples)
    
    # Split
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    
    # Save splits
    train_metadata = {'samples': train_samples}
    val_metadata = {'samples': val_samples}
    
    train_path = os.path.join(output_dir, 'train_metadata.json')
    val_path = os.path.join(output_dir, 'val_metadata.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(val_path, 'w') as f:
        json.dump(val_metadata, f, indent=2)
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Train metadata: {train_path}")
    print(f"Val metadata: {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare CC12M dataset")
    parser.add_argument('--tsv_path', type=str, help='Path to cc12m.tsv file')
    parser.add_argument('--output_dir', type=str, default='data/cc12m', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to download')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of download workers')
    parser.add_argument('--val_ratio', type=float, default=0.05, help='Validation split ratio')
    parser.add_argument('--download_instructions', action='store_true', help='Show download instructions')
    
    args = parser.parse_args()
    
    if args.download_instructions or args.tsv_path is None:
        download_cc12m_metadata(args.output_dir)
        return
    
    # Process TSV and download images
    metadata = process_tsv(
        args.tsv_path,
        args.output_dir,
        max_samples=args.max_samples,
        num_workers=args.num_workers
    )
    
    # Create train/val split
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    create_train_val_split(metadata_path, args.output_dir, val_ratio=args.val_ratio)
    
    print("\nDataset preparation complete!")
    print(f"Data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
