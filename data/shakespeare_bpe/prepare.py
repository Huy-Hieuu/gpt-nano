"""
Prepare Shakespeare dataset with BPE tokenization.

Downloads the input.txt file from the tinyshakespeare dataset,
tokenizes it using tiktoken (GPT-2 BPE), and saves train/val splits.
"""

import os
import pickle
import requests
import numpy as np
import tiktoken

# Configuration
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
download_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def download_shakespeare():
    """Download Shakespeare dataset if not already present."""
    if not os.path.exists(input_file_path):
        print(f"Downloading Shakespeare dataset from {download_url}...")
        response = requests.get(download_url)
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Download complete!")
    else:
        print(f"Shakespeare dataset already exists at {input_file_path}")


def prepare():
    """Prepare the dataset for training."""
    # Download if needed
    download_shakespeare()

    # Load the data
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    print(f"Length of dataset in characters: {len(data):,}")

    # Initialize GPT-2 tokenizer (BPE)
    enc = tiktoken.get_encoding('gpt2')

    # Encode the dataset
    tokens = enc.encode(data)
    print(f"Length of dataset in tokens: {len(tokens):,}")

    # Convert to numpy array and save (use int32 for compatibility with torch)
    tokens = np.array(tokens, dtype=np.int32)

    # Split into train and validation sets (90% / 10%)
    n = len(tokens)
    train_tokens = tokens[:int(n * 0.9)]
    val_tokens = tokens[int(n * 0.9):]

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")

    # Save to binary files
    train_path = os.path.join(os.path.dirname(__file__), 'train.bin')
    val_path = os.path.join(os.path.dirname(__file__), 'val.bin')
    meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')

    # np.save adds .npy extension, so save with temp name then rename
    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)
    # Remove .npy extension
    os.rename(train_path + '.npy', train_path)
    os.rename(val_path + '.npy', val_path)

    # Save metadata (vocab size and encoder name)
    # We don't save encode/decode functions because they can't be pickled
    meta = {
        'vocab_size': enc.n_vocab,
        'encoder_name': 'gpt2',
    }

    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nDataset prepared successfully!")
    print(f"  - Train set: {train_path}")
    print(f"  - Val set: {val_path}")
    print(f"  - Metadata: {meta_path}")
    print(f"  - Vocab size: {meta['vocab_size']}")


if __name__ == '__main__':
    prepare()
