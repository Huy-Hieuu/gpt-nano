"""
Tiny config for quick testing and smoke tests.

Model: ~100K parameters
Training: Very fast, for verification only
"""

# I/O
out_dir = 'out-tiny'
eval_interval = 10
log_interval = 1
eval_iters = 20
wandb_log = False
wandb_project = 'gpt-nano'
wandb_run_name = 'gpt-tiny-test'

# Data
dataset = 'shakespeare_bpe'
batch_size = 12
block_size = 64

# Model
n_layer = 2
n_head = 2
n_embd = 64
dropout = 0.0
bias = False

# Optimizer
learning_rate = 1e-3
max_iters = 100
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0
decay_lr = True
warmup_iters = 10
lr_decay_iters = 100
min_lr = 6e-5

# DDP settings
backend = 'nccl'

# System
device = 'cpu'  # Use CPU for tiny test
compile = False  # Skip compile for tiny test
dtype = 'float32'
