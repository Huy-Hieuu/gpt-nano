# GPT Architecture Reference

Based on the original GPT-2 paper ("Language Models are Unsupervised Multitask Learners") and Andrej Karpathy's nanoGPT.

---

## High-Level Overview

```
Input Text
    │
    ▼
┌──────────┐
│ Tokenizer│   "hello world" → [58, 61, 58, 58, 71, 1, 77, 71, 72, 58, 61]
└──────────┘
    │
    ▼
┌───────────────┐
│ Token + Pos   │   Token Embedding + Positional Embedding
│  Embeddings   │   (learned, not sinusoidal)
└───────────────┘
    │
    ▼
┌───────────────────┐
│  Transformer      │
│  Block × N        │──► repeated N times (e.g. 6 for small, 12 for GPT-2 Small)
│                   │
│  ┌─────────────┐  │
│  │ Layer Norm  │  │
│  ├─────────────┤  │
│  │ Multi-Head  │  │
│  │ Self-Attn   │  │
│  │ (Causal)    │  │
│  ├─────────────┤  │
│  │ Residual    │  │──► x = x + attn(LN(x))
│  ├─────────────┤  │
│  │ Layer Norm  │  │
│  ├─────────────┤  │
│  │ MLP         │  │
│  │ (FFN)       │  │
│  ├─────────────┤  │
│  │ Residual    │  │──► x = x + mlp(LN(x))
│  └─────────────┘  │
└───────────────────┘
    │
    ▼
┌───────────────┐
│  Final Layer  │
│  Norm         │
└───────────────┘
    │
    ▼
┌───────────────┐
│  Linear Head  │──► projects to vocab size (shared weights with token embedding)
│  (to logits)  │
└───────────────┘
    │
    ▼
┌───────────────┐
│  Softmax      │──► probability distribution over vocabulary
└───────────────┘
    │
    ▼
  Output Text (via sampling: greedy, top-k, nucleus)
```

---

## Component Details

### 1. Tokenizer

Converts raw text into integer sequences the model can process.

| Type            | Description                          | Vocab Size        |
| --------------- | ------------------------------------ | ----------------- |
| Character-level | Each character is a token            | ~65 (Shakespeare) |
| BPE (tiktoken)  | Byte Pair Encoding — subword tokens | 50,257 (GPT-2)    |

```
"hello" → character: [h, e, l, l, o] → [46, 43, 50, 50, 53]
"hello" → BPE:       [hel, lo]       → [31373, 275]
```

Character-level is simpler for learning; BPE is more efficient for production.

---

### 2. Token Embedding + Positional Embedding

```python
# Both are learned lookup tables
token_emb = nn.Embedding(vocab_size, n_embd)    # [vocab_size, n_embd]
pos_emb   = nn.Embedding(block_size, n_embd)    # [block_size, n_embd]

# Forward: sum them
x = token_emb(tokens) + pos_emb(positions)      # [batch, block_size, n_embd]
```

- **Token Embedding**: maps each token ID to a dense vector
- **Positional Embedding**: gives the model a sense of token order (learned, NOT sinusoidal like original Transformer)
- `n_embd` = embedding dimension (e.g. 384 for small, 768 for GPT-2 Small)
- `block_size` = max context length (e.g. 256 for char-level, 1024 for GPT-2)

**Key difference from original Transformer**: GPT uses learned positional embeddings instead of fixed sinusoidal encodings.

---

### 3. Transformer Block (repeated N times)

Each block has two sub-layers, each wrapped in a residual connection:

```
x ──────────────────────────(+)────── x_out
│                            ▲
│   ┌───────┐   ┌────────┐  │
└──►│ LN    │──►│ MHSA   │──┘
    └───────┘   └────────┘


x ──────────────────────────(+)────── x_out
│                            ▲
│   ┌───────┐   ┌────────┐  │
└──►│ LN    │──►│ MLP    │──┘
    └───────┘   └────────┘
```

This is the **Pre-LN** variant (LayerNorm before the sublayer), which is what GPT-2 uses. The original Transformer used Post-LN (after), but Pre-LN trains more stably.

---

### 4. Multi-Head Causal Self-Attention

The core mechanism — lets each token "look at" other tokens to gather context.

#### 4a. Single-Head Attention

```python
# For each token, compute Q, K, V
Q = x @ W_q   # [batch, block_size, head_size]
K = x @ W_k   # [batch, block_size, head_size]
V = x @ W_v   # [batch, block_size, head_size]

# Attention scores: how much token i attends to token j
attn = Q @ K.transpose(-2, -1) / sqrt(head_size)   # [batch, block_size, block_size]

# CAUSAL MASK: prevent attending to future tokens
mask = torch.tril(torch.ones(block_size, block_size))  # lower triangular
attn = attn.masked_fill(mask == 0, float('-inf'))

attn = softmax(attn)                                 # normalize rows

# Weighted sum of values
out = attn @ V    # [batch, block_size, head_size]
```

**Why causal masking?** During training, we have the full sequence, but at inference time we generate left-to-right. The mask ensures training and inference are consistent — each token can only see past tokens.

**Why scale by sqrt(head_size)?** Prevents dot products from growing too large (which pushes softmax into saturation → tiny gradients).

#### 4b. Multi-Head Attention

```
          x
          │
    ┌─────┼─────┐─────┐
    ▼     ▼     ▼     ▼
  Head 1 Head 2 Head 3 ... Head h
    │     │     │     │
    └─────┼─────┘─────┘
          │
     Concatenate
          │
       Linear (W_o)
          │
        Output
```

```python
# Multiple heads run in parallel, each learning different attention patterns
# head_size = n_embd // n_head
# e.g. n_embd=384, n_head=6 → head_size=64

heads = [attention_head(x) for _ in range(n_head)]
out = concat(heads) @ W_o   # [batch, block_size, n_embd]
```

Each head learns different relationships: syntax, semantics, positional patterns, etc.

---

### 5. MLP (Feed-Forward Network)

```python
# Two linear layers with GELU activation
x ──► Linear(n_embd, 4*n_embd) ──► GELU ──► Linear(4*n_embd, n_embd)
```

```python
self.c_fc  = nn.Linear(n_embd, 4 * n_embd)   # expand
self.gelu  = nn.GELU()
self.c_proj = nn.Linear(4 * n_embd, n_embd)   # project back
```

- Expansion factor of 4x (e.g. 384 → 1536 → 384)
- **GELU** activation (not ReLU) — smoother, used in GPT-2/BERT
- This is where the model "thinks" / processes the gathered context

---

### 6. Layer Normalization

```python
# Normalize across the embedding dimension
x = (x - mean(x)) / sqrt(var(x) + eps)
x = gamma * x + beta   # learned scale and shift
```

Stabilizes training by keeping activations in a reasonable range. Applied before each sub-layer (Pre-LN).

---

### 7. Output Head

```python
x = ln_f(x)                    # final layer norm
logits = x @ token_emb.weight  # [batch, block_size, vocab_size]
```

**Weight tying**: the output projection reuses the token embedding weights. This reduces parameters and acts as a regularizer.

---

## Data Flow (Full Forward Pass)

```
Input IDs:           [batch, block_size]        e.g. [4, 256]
                         │
Token Embed:         [batch, block_size, 384]
Pos Embed:           [batch, block_size, 384]
x = token + pos:     [batch, block_size, 384]
                         │
                ┌────────┴────────┐
                │  Block 1         │
                │  Block 2         │    × N (e.g. 6)
                │  Block 3         │
                │  ...             │
                └────────┬────────┘
                         │
                [batch, block_size, 384]
                         │
Final LN:            [batch, block_size, 384]
                         │
Output Linear:       [batch, block_size, vocab_size]
                         │
Loss: cross_entropy(logits, targets)
```

---

## Training

### Loss Function

**Cross-entropy loss** (same as negative log-likelihood):

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

Given a sequence `[A, B, C, D]`, the model learns:

- Given A → predict B
- Given A,B → predict C
- Given A,B,C → predict D

All predictions happen in one forward pass (teacher forcing).

### Optimizer: AdamW

```
AdamW = Adam + decoupled weight decay
```

- `weight_decay=0.1` — prevents overfitting
- `betas=(0.9, 0.999)` — momentum parameters
- Learning rate: warmup then cosine decay

### Learning Rate Schedule

```
LR
│     ╱────────────────╲
│    ╱                   ╲
│   ╱                      ╲
│  ╱                         ╲
│ ╱                            ╲
│╱
└──────────────────────────────── Iterations
  warmup          cosine decay
  (e.g. 100 iters)   (to 10% of max LR)
```

### Gradient Accumulation

When GPU memory limits batch size, accumulate gradients over multiple mini-batches:

```python
# effective_batch = batch_size × gradient_accumulation_steps
for micro_step in range(grad_accum_steps):
    loss = model(x, y) / grad_accum_steps
    loss.backward()
optimizer.step()
```

### Mixed Precision

```python
# Use float16/bfloat16 for forward+backward, float32 for optimizer update
# ~2x speedup, ~50% memory reduction
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits, loss = model(x, y)
```

---

## Sampling (Inference)

```python
# Autoregressive generation — one token at a time
context = [start_token]
for _ in range(max_new_tokens):
    logits = model(context[:, -block_size:])   # crop to block_size
    logits = logits[:, -1, :]                  # last timestep
    probs = softmax(logits / temperature)

    # Optional: top-k sampling
    top_k_probs, top_k_idx = probs.topk(k)
    next_token = torch.multinomial(top_k_probs, 1)

    context = torch.cat([context, next_token], dim=1)
```

- **Temperature**: controls randomness (low = deterministic, high = random)
- **Top-k**: only sample from the k most likely tokens (reduces nonsense)
- **Greedy**: just take argmax (repetitive but consistent)

---

## GPT-2 Model Sizes (Reference)

| Model        | n_layer | n_head | n_embd | Parameters |
| ------------ | ------- | ------ | ------ | ---------- |
| GPT-2 Small  | 12      | 12     | 768    | 124M       |
| GPT-2 Medium | 24      | 16     | 1024   | 350M       |
| GPT-2 Large  | 36      | 20     | 1280   | 774M       |
| GPT-2 XL     | 48      | 25     | 1600   | 1.5B       |
| nano (this)  | 6       | 6      | 384    | ~10M       |

---

## Key Design Choices Summary

| Choice              | GPT-2                           | Original Transformer | Why                              |
| ------------------- | ------------------------------- | -------------------- | -------------------------------- |
| Positional encoding | Learned                         | Sinusoidal           | More flexible                    |
| Layer norm position | Pre-LN                          | Post-LN              | More stable training             |
| Activation          | GELU                            | ReLU                 | Smoother gradients               |
| Attention           | Causal (decoder-only)           | Full encoder-decoder | Autoregressive generation        |
| Normalization       | LayerNorm                       | LayerNorm            | —                               |
| Bias in LN/Linear   | Yes (GPT-2) / No (GPT-2 modern) | Yes                  | Removing bias is slightly faster |

---

## File-to-Architecture Mapping

When you implement this, the code maps to components as follows:

| File                  | Component                                            |
| --------------------- | ---------------------------------------------------- |
| `model.py`          | `CausalSelfAttention`, `MLP`, `Block`, `GPT` |
| `train.py`          | Training loop, optimizer, LR schedule, logging       |
| `config/*.py`       | Hyperparameters (n_layer, n_head, n_embd, etc.)      |
| `data/*/prepare.py` | Tokenizer + dataset preparation                      |
| `sample.py`         | Text generation from checkpoint                      |
