# GPT Deep Dive — Learning Guide

This document explains each component of GPT using analogies and step-by-step examples. Read this alongside `architecture.md`.

---

## Table of Contents

1. [What is a Language Model?](#1-what-is-a-language-model)
2. [Tokenizer — Text to Numbers](#2-tokenizer--text-to-numbers)
3. [Embeddings — Numbers to Meaning](#3-embeddings--numbers-to-meaning)
4. [Self-Attention — Tokens Talking to Each Other](#4-self-attention--tokens-talking-to-each-other)
5. [Multi-Head Attention — Multiple Perspectives](#5-multi-head-attention--multiple-perspectives)
6. [MLP — Thinking About What You Heard](#6-mlp--thinking-about-what-you-heard)
7. [Residual Connections — The Highway](#7-residual-connections--the-highway)
8. [Layer Normalization — Keeping Things Calm](#8-layer-normalization--keeping-things-calm)
9. [The Transformer Block — Putting It Together](#9-the-transformer-block--putting-it-together)
10. [Training — How the Model Learns](#10-training--how-the-model-learns)
11. [Sampling — Generating Text](#11-sampling--generating-text)

---

## 1. What is a Language Model?

A language model does one thing: **predict the next token**.

```
"The cat sat on the" → predicts "mat" (not "refrigerator")
```

That's it. There's no magic — just statistical pattern matching at massive scale. If you can predict the next word accurately, you can write essays, code, poetry, anything — because writing IS just predicting one word after another.

**Analogy**: Imagine autocomplete on your phone, but instead of suggesting the next word based on simple frequency, it understands grammar, context, facts, and style.

---

## 2. Tokenizer — Text to Numbers

### The Problem

Neural networks only understand numbers. We need to convert text → numbers.

### Character-Level Tokenizer (simplest)

Each character gets a unique integer ID.

```python
text = "hello"
chars = sorted(set(text))  # ['e', 'h', 'l', 'o']

# Create lookup tables
char_to_id = {'e': 0, 'h': 1, 'l': 2, 'o': 3}
id_to_char = {0: 'e', 1: 'h', 2: 'l', 3: 'o'}

# Encode
encoded = [char_to_id[c] for c in "hello"]  # [1, 0, 2, 2, 3]

# Decode
decoded = ''.join(id_to_char[i] for i in [1, 0, 2, 2, 3])  # "hello"
```

### BPE Tokenizer (what GPT-2 uses)

Instead of single characters, merge frequently occurring pairs into subword units.

```
Step 0:  h e l l o   w o r l d
Step 1:  h e ll o   w o r l d      (merge "ll")
Step 2:  h e ll o   wor l d       (merge "wo" → then "wor")
...
Final:   [hello] [world]           (eventually common words become single tokens)
```

### Why it matters

| Tokenizer | Vocab Size | Pros                         | Cons                             |
| --------- | ---------- | ---------------------------- | -------------------------------- |
| Character | ~65-256    | Simple, no unknown tokens    | Sequences are very long          |
| BPE       | ~50,000    | Shorter sequences, efficient | More complex, needs pre-training |

**Analogy**: Character-level is spelling each word letter by letter. BPE is like reading — you recognize whole words and word parts instantly. Both work, but BPE is faster.

---

## 3. Embeddings — Numbers to Meaning

### The Problem

Token ID `46` is just an integer — it has no meaning. The model needs a rich representation.

### Token Embedding -> What

Each token ID maps to a learned vector (a list of numbers). Similar words end up with similar vectors.

```python
# Imagine n_embd = 4 (real models use 384-1600)
token_emb = {
    "cat":   [0.8, 0.2, 0.9, 0.1],   # animal-like
    "dog":   [0.7, 0.3, 0.8, 0.2],   # animal-like (similar to cat!)
    "king":  [0.9, 0.8, 0.3, 0.1],   # royalty-like
    "queen": [0.8, 0.9, 0.2, 0.1],   # royalty-like (similar to king!)
}
```

These vectors are **learned during training** — the model figures out what dimensions matter.

### Positional Embedding -> Where

The model needs to know WHERE each token is. "dog bites man" ≠ "man bites dog".

```python
# Position 0: [0.1, 0.9, 0.0, 0.3]
# Position 1: [0.5, 0.2, 0.8, 0.1]
# Position 2: [0.3, 0.7, 0.4, 0.6]
# ... (learned, not fixed)

# "cat" at position 2:
x = token_emb["cat"] + pos_emb[2]
  = [0.8, 0.2, 0.9, 0.1] + [0.3, 0.7, 0.4, 0.6]
  = [1.1, 0.9, 1.3, 0.7]
```

### Combined

```python
# For input "The cat sat"
tokens = [46, 23, 58]   # token IDs for T,h,e / c,a,t / s,a,t

x = token_emb[tokens] + pos_emb[[0, 1, 2]]
# x shape: [3, n_embd]  →  3 tokens, each represented by n_embd numbers
```

**Analogy**: Token embedding is like a dictionary definition. Positional embedding is like knowing which word in a sentence you're looking at. Together, you know WHAT you're reading AND WHERE it is.

---

## 4. Self-Attention — Tokens Talking to Each Other

### The Problem

"The **animal** didn't cross the street because **it** was too tired."

What does "it" refer to? The animal or the street? You know it's "animal" from context. The model needs the same ability.

### The Intuition

Self-attention lets every token "look at" every other token and decide **how much to pay attention to each one**.

Imagine you're at a meeting:

- You (token) have a **question** (Query)
- Everyone else has **topics they know about** (Key)
- When someone's topic matches your question, you **listen more** to their **answer** (Value)

### Step-by-Step Example (Fully Traced with Real Numbers)

Let's trace attention for the sentence: **"The cat sat"** using `head_size = 3` (keeping it tiny so we can see every number).

#### Step 0: Start with token embeddings

After token + positional embedding, each token is a vector. For clarity, here are the starting embeddings:

```python
# Shape: [3 tokens, 3 dimensions]  (real models use ~384 dimensions)

x = [
    "The" → [1.0, 0.5, 0.2],    # x₁
    "cat" → [0.3, 0.8, 0.6],    # x₂
    "sat" → [0.1, 0.4, 0.9],    # x₃
]
```

These are the inputs. Now we need to create Q, K, V from each one.

#### Step 1: Create Q, K, V from each token

Each token's embedding `x` gets multiplied by **three learned weight matrices** (W_q, W_k, W_v) to produce three new vectors:

```python
# These weight matrices are learned during training.
# They start random and get better over time.
# Shape: [head_size, head_size] = [3, 3] in our example

W_q = [[0.5, 0.1, 0.3],
       [0.2, 0.7, 0.1],
       [0.4, 0.2, 0.6]]

W_k = [[0.3, 0.5, 0.1],
       [0.6, 0.2, 0.4],
       [0.1, 0.8, 0.3]]

W_v = [[0.1, 0.4, 0.7],
       [0.8, 0.2, 0.3],
       [0.5, 0.6, 0.1]]
```

Now compute Q, K, V for **each** token by multiplying `x @ W`:

```python
# Q = x @ W_q  (query: "what am I looking for?")
# K = x @ W_k  (key:   "what do I contain?")
# V = x @ W_v  (value: "what information do I provide?")

# --- For "The" (x₁ = [1.0, 0.5, 0.2]) ---
Q₁ = [1.0, 0.5, 0.2] @ W_q = [0.63, 0.57, 0.44]
K₁ = [1.0, 0.5, 0.2] @ W_k = [0.62, 0.76, 0.31]
V₁ = [1.0, 0.5, 0.2] @ W_v = [0.60, 0.62, 0.83]  ← this is V₁!

# --- For "cat" (x₂ = [0.3, 0.8, 0.6]) ---
Q₂ = [0.3, 0.8, 0.6] @ W_q = [0.50, 0.65, 0.49]
K₂ = [0.3, 0.8, 0.6] @ W_k = [0.57, 0.67, 0.55]
V₂ = [0.3, 0.8, 0.6] @ W_v = [0.87, 0.62, 0.51]  ← this is V₂!

# --- For "sat" (x₃ = [0.1, 0.4, 0.9]) ---
Q₃ = [0.1, 0.4, 0.9] @ W_q = [0.45, 0.47, 0.59]
K₃ = [0.1, 0.4, 0.9] @ W_k = [0.36, 0.87, 0.46]
V₃ = [0.1, 0.4, 0.9] @ W_v = [0.84, 0.72, 0.34]  ← this is V₃!
```

**So where do V₁, V₂, V₃ come from?**

They come from **multiplying each token's embedding by the learned W_v matrix**. The same input `x` is transformed three different ways:

```
           x (token embedding)
           │
     ┌─────┼─────┐
     │     │     │
   @ W_q  @ W_k  @ W_v
     │     │     │
     Q     K     V
  "what   "what  "what
   am I    do I   info
  looking  have   do I
   for?"   to      carry?"
           show?"
```

Think of it this way:
- **Q (Query)** = the question you're asking about the sentence
- **K (Key)** = the label on your forehead that says what you're about
- **V (Value)** = the actual content / meaning you contribute if someone attends to you

All three come from the **same input**, just transformed differently.

#### Step 2: Compute attention scores

```python
# How much should token i attend to token j?
# score(i,j) = Qᵢ · Kⱼ  (dot product = similarity)

scores = Q @ K.T = [
#       The   cat   sat
    [  0.8,  0.2,  0.1],   # "The" attends to...
    [  0.3,  0.9,  0.4],   # "cat" attends to...
    [  0.2,  0.5,  0.7],   # "sat" attends to...
]
```

"cat" (row 2) has the highest score for itself (0.9) — makes sense, it's a noun with specific meaning. But it also attends to "sat" (0.4) — context matters.

#### Step 3: Scale and mask (causal - Causal means left-to-right only)

The word comes from causality — cause and effect. In time/sequence, the past causes the present, but the
  future cannot influence the past.

  You write a sentence left to right. Word 5 exists because of words 1-4. Word 5 cannot influence words 1-4
  because it hasn't been written yet.

```python
# Scale: divide by sqrt(head_size) to keep values manageable
scores = scores / sqrt(64)  # if head_size=64

# Causal mask: each token can only see itself and PREVIOUS tokens
# (because at inference, future tokens don't exist yet)
mask = [
    [1, 0, 0],   # "The" can see: The only
    [1, 1, 0],   # "cat" can see: The, cat
    [1, 1, 1],   # "sat" can see: The, cat, sat
]

scores = scores.masked_fill(mask == 0, -inf)

# After masking:
scores = [
    [  0.8,  -inf,  -inf],
    [  0.3,   0.9,  -inf],
    [  0.2,   0.5,   0.7],
]
```

**Why `-inf`?** After softmax, `-inf` becomes 0, so those positions get zero attention.

#### Step 4: Softmax → Attention Weights

```python
weights = softmax(scores, dim=-1)

weights = [
    [1.00, 0.00, 0.00],   # "The" only attends to itself (only option)
    [0.35, 0.65, 0.00],   # "cat" mostly attends to itself, some to "The"
    [0.26, 0.32, 0.42],   # "sat" attends to all previous roughly evenly
]
```

Each row sums to 1.0. These are the "attention weights" — how much each token cares about every other token.

#### Step 5: Weighted sum of Values → The Final Output

Now we use the attention weights to **mix** the V vectors. This is where context actually flows between tokens.

```python
# Recall our V vectors from Step 1:
V₁ ("The") = [0.60, 0.62, 0.83]   # "The"'s content
V₂ ("cat") = [0.87, 0.62, 0.51]   # "cat"'s content
V₃ ("sat") = [0.84, 0.72, 0.34]   # "sat"'s content
```

Now multiply the attention weights by the V vectors:

```python
output = weights @ V

# ── For "The" (position 0): ──
# weights[0] = [1.00, 0.00, 0.00]
# Only attends to itself, so:
output[0] = 1.00 * V₁ + 0.00 * V₂ + 0.00 * V₃
          = 1.00 * [0.60, 0.62, 0.83]
          = [0.60, 0.62, 0.83]
# (no context from others — it's the first token, it can only see itself)

# ── For "cat" (position 1): ──
# weights[1] = [0.35, 0.65, 0.00]
# Mostly attends to itself, some to "The", none to "sat":
output[1] = 0.35 * V₁ + 0.65 * V₂ + 0.00 * V₃
          = 0.35 * [0.60, 0.62, 0.83]
          + 0.65 * [0.87, 0.62, 0.51]
          + 0.00 * [0.84, 0.72, 0.34]

          = [0.210, 0.217, 0.291]    ← 35% from "The"
          + [0.566, 0.403, 0.332]    ← 65% from "cat"

          = [0.776, 0.620, 0.623]    ← "cat" now carries info from BOTH tokens!

# ── For "sat" (position 2): ──
# weights[2] = [0.26, 0.32, 0.42]
# Attends to all three tokens:
output[2] = 0.26 * V₁ + 0.32 * V₂ + 0.42 * V₃
          = 0.26 * [0.60, 0.62, 0.83]
          + 0.32 * [0.87, 0.62, 0.51]
          + 0.42 * [0.84, 0.72, 0.34]

          = [0.156, 0.161, 0.216]    ← 26% from "The"
          + [0.278, 0.198, 0.163]    ← 32% from "cat"
          + [0.353, 0.302, 0.143]    ← 42% from "sat"

          = [0.787, 0.661, 0.522]    ← "sat" now carries info from ALL tokens
```

### What just happened?

Before attention, each token only knew about **itself**:

```
"The" → [1.0, 0.5, 0.2]   ← just "The", no context
"cat" → [0.3, 0.8, 0.6]   ← just "cat", no context
"sat" → [0.1, 0.4, 0.9]   ← just "sat", no context
```

After attention, each token carries a **blend of information** from the tokens it attended to:

```
"The" → [0.600, 0.620, 0.830]   ← still just itself (it's first)
"cat" → [0.776, 0.620, 0.623]   ← itself + some "The"
"sat" → [0.787, 0.661, 0.522]   ← itself + "The" + "cat"
```

**This is the core magic of self-attention.** Each token's representation now contains contextual information from other tokens. The more a token "attends" to another, the more of that other token's V (content) it absorbs.

**Analogy**: Imagine 3 people in a room. Each person has a piece of information (V). They ask each other questions (Q) and check if their topics match (K). When they match, they share their information. After the conversation, each person walks away knowing not just their own info, but a weighted blend of everyone else's info too.

---

## 5. Multi-Head Attention — Multiple Perspectives

### The Problem

One set of Q, K, V can only capture one type of relationship. But language has many:

- Grammar: "The **cat** **sat"** (subject-verb)
- Reference: "**The cat** ... **it**" (pronoun resolution)
- Position: "not **very** good" (adjacent modifiers)

### The Solution

Run multiple attention heads in parallel, each learning different patterns.

```
Input x (n_embd = 384)
    │
    ├── Head 1 (head_size = 64)  → might learn grammar patterns
    ├── Head 2 (head_size = 64)  → might learn coreference
    ├── Head 3 (head_size = 64)  → might learn nearby word patterns
    ├── Head 4 (head_size = 64)  → might learn long-range dependencies
    ├── Head 5 (head_size = 64)  → might learn punctuation structure
    └── Head 6 (head_size = 64)  → might learn something else entirely
    │
    Concatenate: 6 × 64 = 384 = n_embd
    │
    Linear projection: 384 → 384
    │
    Output
```

**Analogy**: Imagine 6 people reading the same sentence, each asked a different question:

- Person 1: "Who is doing the action?"
- Person 2: "What is being acted upon?"
- Person 3: "Where is this happening?"
- etc.

Then you combine all their answers.

### Efficient Implementation

In practice, we don't run 6 separate attention operations. We do it in one matrix operation:

```python
# All heads at once
Q = x @ W_q   # [batch, seq_len, n_embd]
K = x @ W_k   # [batch, seq_len, n_embd]
V = x @ W_v   # [batch, seq_len, n_embd]

# Reshape to separate heads
Q = Q.view(batch, seq_len, n_head, head_size).transpose(1, 2)
# Now: [batch, n_head, seq_len, head_size]

# Attention (batch and n_head dimensions are independent)
attn = (Q @ K.transpose(-2, -1)) / sqrt(head_size)
attn = attn.masked_fill(mask == 0, -inf)
attn = softmax(attn)
out = attn @ V   # [batch, n_head, seq_len, head_size]

# Merge heads back
out = out.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)
# [batch, n_head, seq_len, head_size] → [batch, seq_len, n_embd]

# Final projection
out = out @ W_o
```

---

## 6. MLP — Thinking About What You Heard

### The Problem

Attention gathers information from other tokens. But then what? You need to **process** that information.

### The MLP (Multi-Layer Perceptron)

```python
# Simple but effective: expand → activate → contract
x (384) → Linear → (1536) → GELU → Linear → (384)
```

Think of it as: attention is **gathering** information, MLP is **reasoning** about it.

### Why expand 4x?

```python
# The expansion gives the model "room to think"
# Imagine you're solving a math problem:
#   Input:  "2 + 3 = ?"
#   You need workspace to compute:
#   Expanded: "Okay, 2 and 3, those are small numbers,
#              addition means combining them, that gives 5"
#   Output: "5"

# n_embd=384, 4*384=1536
self.c_fc   = nn.Linear(384, 1536)   # expand to working memory
self.gelu   = nn.GELU()              # nonlinear transformation
self.c_proj = nn.Linear(1536, 384)   # compress back to summary
```

### Why GELU instead of ReLU?

```python
# ReLU: max(0, x) — harsh cutoff at zero
# GELU: x * Φ(x)  — smooth, gentle curve

# ReLU:          GELU:
#   ╱                ╱
#  ╱                ╱
# ──────         ───╱───
#                ╱   (smooth transition around 0)
#
# GELU keeps small negative values slightly (unlike ReLU which kills them)
# This makes training smoother for language models
```

**Analogy**: After a group meeting (attention), you go back to your desk and think deeply about what everyone said (MLP). The 4x expansion is like spreading out your notes on a big table before summarizing them.

---

## 7. Residual Connections — The Highway

### The Problem

Deep networks (many layers) are hard to train. Gradients vanish — the learning signal gets weaker as it passes through many layers.

### The Solution: Skip Connections

```python
# Instead of:
x = layer(x)

# Do:
x = x + layer(x)
#   ^     ^
#   |     new transformation
#   original information passes through unchanged
```

### Why This Works

```
Without residual:              With residual:

Input                           Input
  ↓                              ↓
Layer 1                          Layer 1
  ↓                              ↓
Layer 2                          (+) ← + original input
  ↓                              ↓
Layer 3                          Layer 2
  ↓                              ↓
Output                           (+) ← + Layer 1 output
                                 ↓
                                 Layer 3
                                 ↓
                                 (+) ← + Layer 2 output
                                 ↓
                                 Output
```

The gradient can always flow backwards through the `+ x` path without being transformed. It's like having an emergency staircase in a building — even if the elevators (layers) break, you can still get out.

**Analogy**: You're editing a document. Instead of overwriting the original, you add a new paragraph. The original text is always preserved, and each layer just adds refinements on top.

---

## 8. Layer Normalization — Keeping Things Calm

### The Problem

As data flows through many layers, the values can grow or shrink unpredictably. This makes training unstable.

### The Solution

```python
# For each token's embedding, normalize across the embedding dimension:

x = [2.1, -0.5, 3.8, 0.2]   # raw values — all over the place
mean = 1.4
std = 1.7

x_norm = (x - mean) / std
       = [0.41, -1.12, 1.41, -0.71]  # centered around 0, roughly unit scale

# Then learn to scale and shift (let the model decide the best range)
x_out = gamma * x_norm + beta  # gamma and beta are learned
```

### Why Not Batch Normalization?

Batch Norm normalizes across the batch dimension — but:

- It depends on batch size (unstable with small batches)
- It's tricky for autoregressive models (causal masking)
- Layer Norm normalizes per-token, per-position — independent of batch

**Analogy**: Imagine each token's embedding is a student's test scores. Layer Norm converts everything to a standard scale (like grading on a curve), so no one layer's output overwhelms the next.

---

## 9. The Transformer Block — Putting It Together

One transformer block = one "round" of reading + thinking:

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        # Step 1: Attend (gather info from other tokens)
        x = x + self.attn(self.ln_1(x))    # residual + LN + attention

        # Step 2: Think (process the gathered info)
        x = x + self.mlp(self.ln_2(x))     # residual + LN + MLP

        return x
```

### The Order Matters (Pre-LN)

```
Pre-LN (GPT-2):     x = x + sublayer(LN(x))    ← Stable ✓
Post-LN (Original): x = LN(x + sublayer(x))     ← Less stable ✗
```

Pre-LN applies normalization *before* the sublayer. This ensures the input to attention/MLP is always well-behaved, which stabilizes training especially in deep networks.

### Full GPT: Stack N Blocks

```python
class GPT(nn.Module):
    def __init__(self, config):
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: output head shares weights with token embedding
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape

        tok = self.tok_emb(idx)              # [B, T, n_embd]
        pos = self.pos_emb(torch.arange(T))  # [T, n_embd]
        x = tok + pos                        # [B, T, n_embd]

        for block in self.blocks:
            x = block(x)                     # [B, T, n_embd]

        x = self.ln_f(x)                     # [B, T, n_embd]
        logits = self.head(x)                # [B, T, vocab_size]

        return logits
```

---

## 10. Training — How the Model Learns

### The Setup

Given a sequence `[A, B, C, D, E]`, we create input-target pairs:

```
Input:    A     → predict B
Input:    A B   → predict C
Input:    A B C → predict D
Input:    A B C D → predict E
```

But we do ALL of these in **one forward pass** (this is the magic of causal masking):

```python
# Input:  [A, B, C, D]
# Target: [B, C, D, E]
#                ↑  ↑  ↑  ↑
#                These are shifted by one position

logits = model([A, B, C, D])   # [1, 4, vocab_size]

# Compare each position's prediction to the target
loss = cross_entropy(logits, [B, C, D, E])
```

### Cross-Entropy Loss (Intuition)

```python
# The model outputs probabilities for every token in the vocabulary
# If vocab_size = 65 (Shakespeare chars), it outputs 65 probabilities

# Example for position predicting "e":
# Model predicts: {'a': 0.01, 'b': 0.02, ..., 'e': 0.60, ..., 'z': 0.01}
# Target: 'e'
# Loss = -log(0.60) = 0.51  (lower is better)

# If model predicts 'e' with probability 0.99:
# Loss = -log(0.99) = 0.01  (very good!)

# If model predicts 'e' with probability 0.01:
# Loss = -log(0.01) = 4.61  (very bad!)
```

### How Gradients Work (Simplified)

```
1. Forward pass: compute predictions and loss
2. Backward pass: compute gradients (how much each parameter affected the loss)
3. Update: nudge each parameter to reduce loss

# Parameter update (simplified):
weight = weight - learning_rate * gradient
#                        ↑              ↑
#                   how big a step   which direction
```

### Learning Rate Schedule

```python
# Don't use the same LR throughout:

# Phase 1: Warmup (first ~100 steps)
# Start with very small LR → gradually increase
# Why? Random initial weights produce random gradients
#      Large LR * random gradients = unstable training

# Phase 2: Cosine Decay (rest of training)
# Slowly decrease LR from max to ~10% of max
# Why? Early on, big steps help explore the loss landscape
#      Later, small steps help fine-tune the solution

# Visual:
# LR
#  │    ╱‾‾‾‾‾╲
#  │   ╱       ╲
#  │  ╱          ╲
#  │ ╱             ╲
#  │╱
#  └──────────────────→ steps
#   warmup  cosine decay
```

### Gradient Accumulation (When GPU Memory is Tight)

```python
# Problem: Want batch_size=64, but GPU only fits batch_size=8
# Solution: accumulate gradients over 8 mini-batches, then update

effective_batch = 64
micro_batch = 8
accum_steps = effective_batch // micro_batch  # = 8

optimizer.zero_grad()
for i in range(accum_steps):
    x, y = get_batch(micro_batch)
    loss = model(x, y) / accum_steps   # scale loss
    loss.backward()                     # gradients accumulate

optimizer.step()  # one update with accumulated gradients
```

---

## 11. Sampling — Generating Text

### Greedy Decoding

Always pick the most likely next token.

```python
# Simple but boring — tends to repeat itself
next_token = argmax(logits)
```

Example output: "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."

### Temperature

Control randomness by scaling logits before softmax.

```python
# Temperature = 1.0: normal (use learned probabilities)
# Temperature < 1.0: sharper (more confident, more repetitive)
# Temperature > 1.0: flatter (more random, more creative)

logits = logits / temperature
probs = softmax(logits)
```

```
Temperature = 0.5: "The cat sat on the mat and looked at the wall."    (safe)
Temperature = 1.0: "The cat sat on the mat and dreamed of catching."  (balanced)
Temperature = 1.5: "The cat sat on the mat and quantum-physics banana" (chaotic)
```

### Top-k Sampling

Only consider the k most likely tokens, zero out the rest.

```python
# top_k = 50: only sample from the top 50 candidates
# This prevents truly bizarre tokens from being selected

top_logits, top_indices = torch.topk(logits, k=50)
probs = softmax(top_logits)
next_token = top_indices[torch.multinomial(probs, 1)]
```

### Full Generation Loop

```python
def generate(model, start_tokens, max_new_tokens, temperature=0.8, top_k=40):
    idx = start_tokens  # e.g. tensor([[0]]) for newline

    for _ in range(max_new_tokens):
        # Crop to block_size (model can't handle longer sequences)
        idx_cond = idx[:, -block_size:]

        # Forward pass
        logits = model(idx_cond)
        logits = logits[:, -1, :]  # only care about last position

        # Temperature
        logits = logits / temperature

        # Top-k
        if top_k is not None:
            top_vals, _ = torch.topk(logits, top_k)
            logits[logits < top_vals[:, -1:]] = float('-inf')

        # Sample
        probs = softmax(logits)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        idx = torch.cat([idx, next_token], dim=1)

    return idx
```

**Analogy for the whole process**:

1. You write the first word (start token)
2. The model suggests what word could come next
3. You pick one (with some randomness)
4. Repeat until you have a full story

---

## Quick Reference: Dimensions

```
batch_size (B)      = number of sequences processed together
block_size (T)      = max sequence length (context window)
n_embd (C)          = embedding dimension (information capacity per token)
n_head (H)          = number of attention heads
head_size (C/H)     = dimension per attention head
n_layer (N)         = number of transformer blocks (depth)
vocab_size (V)      = number of possible tokens

Parameter count ≈ the sum of:
  - Token embedding:    V × C
  - Position embedding: T × C
  - Per block:
    - Attention:        4 × C²  (Q, K, V, O projections)
    - MLP:              2 × C × 4C = 8C²
    - Layer norms:      ~4C (negligible)
  - Output head:        V × C (if not weight-tied, else 0)

Total ≈ V×C + T×C + N × (12C²)
```

For the nano config (C=384, N=6, V=65, T=256):

```
  65×384 + 256×384 + 6×(12×384²) ≈ 10.7M parameters
```
