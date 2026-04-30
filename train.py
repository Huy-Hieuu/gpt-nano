"""
Training script for GPT Nano.

Features:
- Configurable via config files
- WandB logging
- Mixed precision (bfloat16) support
- Gradient accumulation
- Checkpoint saving and resuming
- Learning rate warmup + cosine decay
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import wandb

import gpt_model as model


# Default configuration (can be overridden by config file)
default_config = dict(
    # I/O
    out_dir='out',
    eval_interval=250,
    log_interval=10,
    eval_iters=200,
    wandb_log=False,
    wandb_project='gpt-nano',
    wandb_run_name='gpt-run',

    # Data
    dataset='shakespeare_bpe',
    batch_size=64,
    block_size=256,

    # Model
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.0,
    bias=False,

    # Optimizer
    learning_rate=3e-4,
    max_iters=5000,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.999,
    grad_clip=1.0,
    decay_lr=True,
    warmup_iters=100,
    lr_decay_iters=5000,
    min_lr=6e-5,

    # DDP settings
    backend='nccl',

    # System
    device='cuda',
    compile=True,
    dtype='bfloat16',
)


def get_lr(iter, config):
    """Learning rate with warmup and cosine decay."""
    if not config['decay_lr']:
        return config['learning_rate']

    # Linear warmup
    if iter < config['warmup_iters']:
        return config['learning_rate'] * iter / config['warmup_iters']

    # Cosine decay
    if iter > config['lr_decay_iters']:
        return config['min_lr']

    decay_ratio = (iter - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])


def get_batch(split, data_dir, config):
    """Get a batch of data."""
    # Load the data
    data = np.load(os.path.join(data_dir, f'{split}.bin'))
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    # Convert tensor indices to numpy array for indexing
    ix_np = ix.numpy()
    x = torch.from_numpy(np.array([data[i:i+config['block_size']] for i in ix_np]))
    y = torch.from_numpy(np.array([data[i+1:i+1+config['block_size']] for i in ix_np]))
    # Convert to long type for embedding and cross_entropy
    x = x.long().to(config['device'])
    y = y.long().to(config['device'])
    return x, y


@torch.no_grad()
def estimate_loss(gpt_model, data_dir, config):
    """Estimate loss on train and val sets."""
    out = {}
    gpt_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(split, data_dir, config)
            logits, loss = gpt_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    gpt_model.train()
    return out


def load_config(config_path):
    """Load config from a Python file."""
    config = default_config.copy()

    # Create a namespace for the config file
    namespace = {}
    with open(config_path, 'r') as f:
        exec(f.read(), namespace)

    # Update config with values from file
    for key in config:
        if key in namespace:
            config[key] = namespace[key]

    return config


def main():
    # Parse command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--device', type=str, default=None, help='Override device')
    parser.add_argument('--compile', action='store_true', default=None, help='Enable torch.compile')
    parser.add_argument('--no-compile', action='store_true', default=None, help='Disable torch.compile')
    parser.add_argument('--dtype', type=str, default=None, help='Override dtype')
    parser.add_argument('--init_from', type=str, default='scratch',
                        help="'scratch' or 'resume'")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config from command line
    if args.device:
        config['device'] = args.device
    if args.compile:
        config['compile'] = True
    elif args.no_compile:
        config['compile'] = False
    if args.dtype:
        config['dtype'] = args.dtype

    # Device setup
    device_type = 'cuda' if 'cuda' in config['device'] else 'cpu'
    torch_device = torch.device(config['device'])

    # Set dtype for autocast
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Data directory
    data_dir = os.path.join('data', config['dataset'])

    # Load metadata (vocab size, encode/decode)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta = pickle.load(open(meta_path, 'rb'))
    vocab_size = meta['vocab_size']
    print(f"Vocab size: {vocab_size}")

    # Create model
    model_config = model.GPTConfig(
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        block_size=config['block_size'],
        vocab_size=vocab_size,
        dropout=config['dropout'],
        bias=config['bias'],
    )
    gpt_model = model.GPT(model_config)
    gpt_model.to(torch_device)

    # Compile model (if requested and supported)
    if config['compile']:
        print("Compiling model...")
        gpt_model = torch.compile(gpt_model)  # type: ignore

    # Optimizer
    optimizer = gpt_model.configure_optimizers(
        weight_decay=config['weight_decay'],
        learning_rate=config['learning_rate'],
        betas=(config['beta1'], config['beta2']),
        device_type=device_type,
    )

    # Setup WandB
    if config['wandb_log']:
        wandb.init(
            project=config['wandb_project'],
            name=config['wandb_run_name'],
            config=config,
        )

    # Resume from checkpoint
    iter_num = 0
    best_val_loss = 1e9

    if args.init_from == 'resume':
        ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            gpt_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resumed from checkpoint at iteration {iter_num}")

    # Create output directory
    os.makedirs(config['out_dir'], exist_ok=True)

    # Training loop
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    X, Y = get_batch('train', data_dir, config)

    while True:
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate the loss on train/val sets and write checkpoints
        if iter_num % config['eval_interval'] == 0 and iter_num > 0:
            losses = estimate_loss(gpt_model, data_dir, config)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if config['wandb_log']:
                wandb.log({
                    'iter': iter_num,
                    'train/loss': losses['train'],
                    'val/loss': losses['val'],
                    'lr': lr,
                    'mfu': running_mfu * 100,  # convert to percentage
                })

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': gpt_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))

        if iter_num == 0 and config['eval_only']:
            break

        # Forward backward update
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits, loss = gpt_model(X, Y)
            loss = loss / config['gradient_accumulation_steps']  # scale loss

        # Backward
        loss.backward()

        # Gradient clipping
        if config['grad_clip'] != 0.0:
            torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), config['grad_clip'])

        # Update
        optimizer.step()

        # Update batch
        X, Y = get_batch('train', data_dir, config)

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config['log_interval'] == 0:
            lossf = loss.item() * config['gradient_accumulation_steps']
            if config['wandb_log']:
                wandb.log({
                    'iter': iter_num,
                    'train/loss': lossf,
                    'lr': lr,
                })
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

        iter_num += 1
        local_iter_num += 1

        # Termination conditions
        if iter_num > config['max_iters']:
            break

    if config['wandb_log']:
        wandb.finish()


# Add method to GPT class for optimizer configuration
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    """Configure AdamW optimizer with weight decay only on certain parameters."""
    # Separate out all parameters to those that will and won't experience weight decay
    decay_params = []
    no_decay_params = []

    for pn, p in self.named_parameters():
        if p.ndim < 2:  # biases and layernorm weights have 1 dimension
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    # Create two parameter groups
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    # Create AdamW optimizer
    # Check if fused AdamW is available (CUDA only)
    try:
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    except:
        fused_available = False
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer


# Monkey patch the method onto the GPT class
model.GPT.configure_optimizers = configure_optimizers

# Add gradient_accumulation_steps to default config
default_config['gradient_accumulation_steps'] = 1
default_config['eval_only'] = False


if __name__ == '__main__':
    main()
