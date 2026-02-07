# Chunked Cross-Entropy (CCE) for MLX

## What is CCE?

CCE (Chunked Cross-Entropy) computes cross-entropy loss without materializing the full logits tensor. This is critical for training LLMs with large vocabularies on memory-constrained devices.

### The Problem

Standard cross-entropy requires computing:
```python
logits = hidden @ weight.T  # Shape: [N, V] where V = vocab size
loss = cross_entropy(logits, targets)
```

For a typical LLM training step:
- `N = batch_size × seq_len = 8 × 2048 = 16,384`
- `V = vocab_size = 128,000` (Llama) or `256,000` (Gemma)
- Memory for logits: `16,384 × 128,000 × 4 bytes = 8 GB`

This single tensor can exceed available memory on Apple Silicon devices.

### The Solution

CCE processes the vocabulary in chunks, never materializing the full `[N, V]` tensor:

```
Forward pass:
  for each chunk of vocab [v_start : v_end]:
    logits_chunk = hidden @ weight[v_start:v_end].T   # [N, chunk_size]
    accumulate logsumexp and target logits

  loss = target_logit - logsumexp

Backward pass:
  for each chunk of vocab [v_start : v_end]:
    logits_chunk = hidden @ weight[v_start:v_end].T
    d_logits = softmax(logits_chunk) - one_hot(targets)
    grad_hidden += d_logits @ weight[v_start:v_end]
    grad_weight[v_start:v_end] = d_logits.T @ hidden
```

Memory is reduced from `O(N × V)` to `O(N × chunk_size)` where `chunk_size ≈ 16K`.

## Usage

```python
import mlx.core as mx

# Inputs
hidden = mx.array(...)   # [batch, seq, hidden_dim] or [N, hidden_dim]
weight = mx.array(...)   # [vocab_size, hidden_dim]
targets = mx.array(...)  # [batch, seq] or [N]

# CCE loss (replaces: logits = hidden @ weight.T; loss = cross_entropy(logits, targets))
loss = mx.fast.cce_loss(
    hidden.reshape(-1, hidden.shape[-1]),  # Flatten to [N, H]
    weight,
    targets.reshape(-1),                    # Flatten to [N]
)

# With gradient computation
def loss_fn(hidden, weight):
    B, S, H = hidden.shape
    return mx.mean(mx.fast.cce_loss(
        hidden.reshape(B * S, H),
        weight,
        targets.reshape(B * S),
    ))

loss, (grad_h, grad_w) = mx.value_and_grad(loss_fn, argnums=(0, 1))(hidden, weight)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden` | array | required | Hidden states `[N, H]` or `[B, S, H]` |
| `weight` | array | required | LM head weight `[V, H]` |
| `targets` | array | required | Target token IDs `[N]` or `[B, S]` |
| `ignore_index` | int | -100 | Token ID to ignore in loss |
| `logit_softcap` | float | 0.0 | Soft cap for logits (used by Gemma-2) |

## Supported Dtypes

- `float32`
- `float16`
- `bfloat16`

Hidden and weight must have matching dtypes.

## How It Works

### Forward Pass

1. **Initialize** running max, sum_exp, and target_logit accumulators
2. **For each vocab chunk:**
   - Compute `logits_chunk = hidden @ weight_chunk.T`
   - Update running logsumexp using numerically stable online algorithm
   - Extract target logits where `target ∈ [v_start, v_end)`
3. **Finalize:** `loss = -target_logit + logsumexp`

### Backward Pass

1. **Retrieve or recompute** logsumexp from forward pass
2. **For each vocab chunk:**
   - Recompute `logits_chunk = hidden @ weight_chunk.T`
   - Compute `d_logits = grad_output × (softmax(logits) - one_hot(target))`
   - Accumulate `grad_hidden += d_logits @ weight_chunk`
   - Compute `grad_weight_chunk = d_logits.T @ hidden`

The backward pass recomputes logits (gradient checkpointing trade-off) to avoid storing the full `[N, V]` tensor.

### Adaptive Chunking

Chunk size is automatically adjusted based on available system memory:
- Default: 16K vocab positions per chunk
- Minimum: 1K (for very large N)
- Aligned to 256 for optimal GPU performance
