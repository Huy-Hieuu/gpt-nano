"""
Text generation script for GPT Nano.

Load a trained checkpoint and generate text.
"""

import os
import pickle
import argparse
import tiktoken

import torch
import torch.nn.functional as F

import gpt_model as model


def sample(
    model,
    start_idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    device='cpu',
):
    """Generate tokens autoregressively."""
    model.eval()

    idx = start_idx.clone().detach()
    idx = idx.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, crop it at block_size
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

            # Forward the model to get the logits for the index in the sequence
            logits, _ = model(idx_cond)

            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='out-shakespeare-bpe',
                        help='Output directory with checkpoint')
    parser.add_argument('--start', type=str, default='\n',
                        help='Starting token (default: newline)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=500,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (lower = more deterministic)')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling (0 = disabled)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu/cuda)')
    args = parser.parse_args()

    # Load checkpoint
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Load config from checkpoint
    ckpt_config = checkpoint['config']

    # Load metadata (for vocab size)
    data_dir = os.path.join('data', ckpt_config['dataset'])
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    vocab_size = meta['vocab_size']
    encoder_name = meta.get('encoder_name', 'gpt2')

    # Initialize encoder/decoder
    enc = tiktoken.get_encoding(encoder_name)
    encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
    decode = lambda l: enc.decode(l)

    # Create model
    model_config = model.GPTConfig(
        n_layer=ckpt_config['n_layer'],
        n_head=ckpt_config['n_head'],
        n_embd=ckpt_config['n_embd'],
        block_size=ckpt_config['block_size'],
        vocab_size=vocab_size,
        dropout=ckpt_config['dropout'],
        bias=ckpt_config['bias'],
    )

    gpt = model.GPT(model_config)
    gpt.load_state_dict(checkpoint['model'])
    gpt.to(args.device)

    # Compile model if available
    if hasattr(torch, 'compile'):
        print("Compiling model...")
        gpt = torch.compile(gpt)

    print(f"Model loaded from iteration {checkpoint['iter_num']}, best val loss: {checkpoint['best_val_loss']:.4f}")

    # Encode the start token
    start_tokens = encode(args.start)
    start_idx = torch.tensor([start_tokens], dtype=torch.long)

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples with temperature={args.temperature}, top_k={args.top_k}\n")
    print("=" * 80)

    for k in range(args.num_samples):
        idx = sample(
            gpt,
            start_idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
        decoded = decode(idx[0].tolist())
        print(f"Sample {k+1}:")
        print(decoded)
        print("=" * 80)


if __name__ == '__main__':
    main()
