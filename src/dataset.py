"""
CC12M Dataset for TinyVLM Training
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class CC12MDataset(Dataset):
    """
    CC12M dataset for vision-language model training
    """

    def __init__(self, metadata_path, image_size=224, tokenizer=None, max_text_length=128):
        """
        Args:
            metadata_path: path to metadata JSON file
            image_size: target image size
            tokenizer: text tokenizer (simple character-level if None)
            max_text_length: maximum text sequence length
        """
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.samples = self.metadata['samples']
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

        # Image transforms
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image_path = sample['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
        except Exception as e:
            # Fallback to random image if loading fails
            print(f"Error loading image {image_path}: {e}")
            image = torch.randn(3, self.image_size, self.image_size)

        # Tokenize caption
        caption = sample['caption']
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption, max_length=self.max_text_length, truncation=True)
            token_ids = torch.tensor(tokens['input_ids'])
        else:
            # Simple character-level tokenization
            token_ids = self._char_tokenize(caption)

        return {
            'image': image,
            'token_ids': token_ids,
            'caption': caption
        }

    def _char_tokenize(self, text):
        """Simple character-level tokenization"""
        # Convert to lowercase and take first max_text_length chars
        text = text.lower()[:self.max_text_length]
        # Convert to IDs (a=1, b=2, ..., space=27, etc.)
        token_ids = [ord(c) - ord('a') + 1 if c.isalpha()
                     else 27 for c in text]
        # Pad to max length
        token_ids = token_ids + [0] * (self.max_text_length - len(token_ids))
        return torch.tensor(token_ids[:self.max_text_length])

    @staticmethod
    def collate_fn(batch):
        """Custom collate function"""
        images = torch.stack([item['image'] for item in batch])

        # Pad token IDs to same length
        max_len = max(item['token_ids'].size(0) for item in batch)
        token_ids = torch.stack([
            torch.cat([item['token_ids'], torch.zeros(
                max_len - item['token_ids'].size(0), dtype=torch.long)])
            for item in batch
        ])

        captions = [item['caption'] for item in batch]

        return {
            'images': images,
            'token_ids': token_ids,
            'captions': captions
        }



