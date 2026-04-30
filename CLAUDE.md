# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Implement a GPT from scratch (nano scale), following Andrej Karpathy's nanoGPT approach. Training runs on Google Colab (connected via VS Code remote SSH), inference locally.

## Architecture

The model follows the original GPT-2 architecture:
- **Tokenizer**: character-level or BPE (tiktoken)
- **Model**: `model.py` — Transformer with multi-head self-attention, MLP blocks, layer norm
- **Trainer**: `train.py` — training loop with gradient accumulation, cosine LR schedule, AdamW
- **Data**: `data/` — raw text corpus + tokenized binary files (`train.bin`, `val.bin`)
- **Config**: `config/` — Python config files for hyperparameters (train_gpt2.py style)
- **Sampling**: `sample.py` — generate text from a checkpoint

## Key Commands

```bash
# Prepare dataset (e.g. Shakespeare)
python data/shakespeare_char/prepare.py

# Train (local, small scale)
python train.py config/train_shakespeare_char.py

# Train on Colab (run from Colab terminal or VS Code remote)
python train.py config/train_gpt2.py --device=cuda

# Sample from checkpoint
python sample.py --out_dir=out-shakespeare-char

# Run a quick smoke test (tiny model, few iterations)
python train.py config/train_shakespeare_char.py \
  --max_iters=10 --eval_interval=5 --n_layer=2 --n_head=2 --n_embd=64
```

## Google Colab + VS Code Setup

- Colab now supports VS Code via the "Connect to a local runtime" or SSH tunnel (e.g. `cloudflared` or `ngrok` approach)
- Mount Google Drive for persistent checkpoints: `from google.colab import drive; drive.mount('/content/drive')`
- Point `out_dir` in config to `/content/drive/MyDrive/gpt-nano/out` for checkpoint persistence across sessions
- Use `wandb` or manual logging to track loss curves across Colab sessions

## Training Tips

- `block_size`: sequence length (context window), start with 256 for char-level
- `batch_size` × `gradient_accumulation_steps` = effective batch size (target ~0.5M tokens/batch for GPT-2 scale)
- Use `torch.compile()` (PyTorch 2.x) for ~2× speedup on Colab A100/T4
- Mixed precision: `dtype = 'bfloat16'` on A100, `'float16'` on T4

## File Conventions

- Config files in `config/` override defaults defined at the top of `train.py`
- Checkpoints saved as `ckpt.pt` containing `model`, `optimizer`, `config`, `iter_num`, `best_val_loss`
- Resume training by setting `init_from = 'resume'` in config
