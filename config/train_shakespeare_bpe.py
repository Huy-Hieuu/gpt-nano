"""
Main config for training on Shakespeare dataset with BPE tokenization.

Model: ~10M parameters
Features: WandB logging, Mixed Precision (bfloat16), gradient accumulation
"""

# I/O
out_dir = 'out-shakespeare-bpe'
eval_interval = 250
log_interval = 10
eval_iters = 200
wandb_log = True
wandb_project = 'gpt-nano'
wandb_run_name = 'gpt-shakespeare-bpe'

# Data
dataset = 'shakespeare_bpe'
batch_size = 64
block_size = 256

# Model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

# Optimizer
learning_rate = 3e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = learning_rate * 0.1

# DDP settings
backend = 'nccl'

# System
device = 'cuda'  # Use GPU for training
compile = True  # Use torch.compile for speedup
dtype = 'bfloat16'  # Use mixed precision for speed
