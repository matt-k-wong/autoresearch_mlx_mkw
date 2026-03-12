"""
Apple Mac OS X MLX Autoresearch
Architecture: Qwen3.5-0.8B (True Hybrid: 3:1 Gated DeltaNet to Gated Attention)
Dataset: TinyStories
"""

import os
import gc
import math
import time
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Qwen3.5-0.8B Architecture Specifications
# ---------------------------------------------------------------------------

@dataclass
class QwenConfig:
    sequence_len: int = 512
    vocab_size: int = 8192
    n_layer: int = 24
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 1536
    rms_norm_eps: float = 1e-6

def apply_rope(x, cos, sin):
    """Applies Rotary Position Embedding."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)

# ---------------------------------------------------------------------------
# Core Layers
# ---------------------------------------------------------------------------

class GeGLU(nn.Module):
    """Experiment: Swap SwiGLU to GeGLU."""
    def __init__(self, config):
        super().__init__()
        # Hidden size is usually 8/3 of the embedding size in Qwen
        hidden_dim = int(8 * config.n_embd / 3)
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))

def chunkwise_linear_attention(q: mx.array, k: mx.array, v: mx.array, chunk_size: int = 128):
    """
    Fuses the OOM-causing outer product into Block-wise Matrix Multiplications.
    Mathematically identical to: (q[..., None, :] @ mx.cumsum(k[..., :, None] * v[..., None, :], axis=1)).squeeze(-2)
    """
    B, T, H_q, D = q.shape
    _, _, H_kv, _ = k.shape
    C = chunk_size
    
    # Pad sequence to a multiple of chunk_size if necessary
    pad_len = (C - (T % C)) % C
    if pad_len > 0:
        q = mx.pad(q, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        
    N = (T + pad_len) // C
    
    # Reshape KV sequence into chunks: (B, H_kv, N, C, D)
    k_c = k.transpose(0, 2, 1, 3).reshape(B, H_kv, N, C, D)
    v_c = v.transpose(0, 2, 1, 3).reshape(B, H_kv, N, C, D)
    
    # ----------------------------------------------------
    # 1. Inter-chunk (Global Block-State) 
    # ----------------------------------------------------
    chunk_state = k_c.transpose(0, 1, 2, 4, 3) @ v_c
    state_inclusive = mx.cumsum(chunk_state, axis=2)
    
    zeros = mx.zeros((B, H_kv, 1, D, D), dtype=chunk_state.dtype)
    S_prev = mx.concatenate([zeros, state_inclusive[:, :, :-1]], axis=2)

    if H_q != H_kv:
        repeats = H_q // H_kv
        S_prev = mx.repeat(S_prev, repeats, axis=1)
        k_c = mx.repeat(k_c, repeats, axis=1)
        v_c = mx.repeat(v_c, repeats, axis=1)
        
    q_c = q.transpose(0, 2, 1, 3).reshape(B, H_q, N, C, D)
    out_inter = q_c @ S_prev  # (B, H_q, N, C, D)
    
    # ----------------------------------------------------
    # 2. Intra-chunk (Local) Attention
    # ----------------------------------------------------
    attn_local = q_c @ k_c.transpose(0, 1, 2, 4, 3)  
    
    i = mx.arange(C)[:, None]
    j = mx.arange(C)[None, :]
    mask = i >= j
    
    attn_local = mx.where(mask, attn_local, 0.0)
    out_intra = attn_local @ v_c  # (B, H_q, N, C, D)
    
    # ----------------------------------------------------
    # 3. Combine and unchunk
    # ----------------------------------------------------
    out = out_intra + out_inter
    out = out.reshape(B, H_q, T + pad_len, D).transpose(0, 2, 1, 3)
    
    if pad_len > 0:
        out = out[:, :T, :, :]
        
    return out

class GatedLinearAttention(nn.Module):
    """
    Gated DeltaNet-style Linear Attention.
    O(N) memory complexity, avoids T x T matrix via cumulative sums.
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.gate = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def __call__(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Linear Attention requires non-negative keys and queries
        q = mx.maximum(q, 0.0) + 1e-5
        k = mx.maximum(k, 0.0) + 1e-5

        # OOM-Proof Chunkwise Linear Attention
        num = chunkwise_linear_attention(q, k, v, chunk_size=128)
        
        # Denominator remains a simple O(T*D) accumulation
        k_cum = mx.cumsum(k, axis=1)
        if self.n_head != self.n_kv_head:
            repeats = self.n_head // self.n_kv_head
            k_cum = mx.repeat(k_cum, repeats, axis=2)
            
        den = mx.sum(q * k_cum, axis=-1, keepdims=True) + 1e-5
        
        y = (num / den).reshape(B, T, C)
        
        # Apply Qwen Gating
        g = mx.sigmoid(self.gate(x))
        return self.o_proj(y * g)

class GatedFullAttention(nn.Module):
    """Standard causal attention with Qwen Gating."""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.gate = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def __call__(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.n_kv_head, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(q.dtype)
        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        g = mx.sigmoid(self.gate(x))
        return self.o_proj(y * g)

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # The 3:1 Hybrid Ratio (Linear : Full)
        if layer_idx % 4 == 3:
            self.attn = GatedFullAttention(config)
        else:
            self.attn = GatedLinearAttention(config)
            
        self.mlp = GeGLU(config)
        self.ln1 = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.ln2 = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)

    def __call__(self, x, cos_sin):
        cos, sin = cos_sin
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

class QwenHybrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Register blocks directly for MLX
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        
        self.norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        head_dim = config.n_embd // config.n_head
        self.cos, self.sin = self._precompute_rope(MAX_SEQ_LEN, head_dim)

    def _precompute_rope(self, T, head_dim, base=10000):
        dim = head_dim // 2
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, dtype=mx.float32) / dim))
        t = mx.arange(T, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        cos = mx.cos(freqs)[None, :, None, :]
        sin = mx.sin(freqs)[None, :, None, :]
        return cos, sin

    def __call__(self, idx):
        B, T = idx.shape
        cos_sin = (self.cos[:, :T, :, :], self.sin[:, :T, :, :])
        
        x = self.embed_tokens(idx)
        for block in self.blocks:
            x = block(x, cos_sin)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

from typing import Union, Callable

class MuonAdamW(optim.Optimizer):
    """
    Hybrid Optimizer for Apple Mac OS X MLX: 
    - Applies Muon (SGD Momentum + Newton-Schulz 5) to >= 2D block parameters.
    - Falls back to AdamW for < 2D parameters (Biases, RMSNorm) and embeddings.
    """
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        muon_lr: Union[float, Callable] = 0.02,
        momentum: float = 0.95,
        betas: tuple = (0.9, 0.95),
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        eps: float = 1e-8,
        vocab_size: int = 8192
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.muon_lr = muon_lr
        self.momentum = momentum
        self.betas = betas
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.eps = eps
        self.vocab_size = vocab_size

    @property
    def get_muon_lr(self):
        return self.muon_lr(self.state["step"]) if callable(self.muon_lr) else self.muon_lr

    @property
    def get_adam_lr(self):
        return self.learning_rate(self.state["step"]) if callable(self.learning_rate) else self.learning_rate

    def init_single(self, parameter: mx.array, state: dict):
        # Isolate internal MLP/Attention matrices for Muon
        if parameter.ndim >= 2 and parameter.shape[0] != self.vocab_size and parameter.shape[1] != self.vocab_size:
            state["momentum_buffer"] = mx.zeros_like(parameter)
        else:
            state["m"] = mx.zeros_like(parameter)
            state["v"] = mx.zeros_like(parameter)
            state["step"] = mx.array(0, dtype=mx.uint32)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        if parameter.ndim >= 2 and parameter.shape[0] != self.vocab_size and parameter.shape[1] != self.vocab_size:
            # ----------------- MUON PATH -----------------
            lr = self.get_muon_lr
            lr = lr.astype(gradient.dtype) if hasattr(lr, 'astype') else mx.array(lr, dtype=gradient.dtype)
            
            if self.weight_decay > 0.0:
                parameter = parameter - lr * self.weight_decay * parameter
            
            buf = state["momentum_buffer"]
            buf = self.momentum * buf + gradient
            state["momentum_buffer"] = buf
            
            orig_shape = parameter.shape
            X = buf.reshape(orig_shape[0], -1)
            X = X.astype(mx.bfloat16)
            X = X / (mx.linalg.norm(X) + 1e-7)
            
            transposed = X.shape[0] > X.shape[1]
            if transposed:
                X = X.T
                
            a, b, c = 3.4445, -4.7750, 2.0315
            for _ in range(self.ns_steps):
                A = X @ X.T
                B = b * A + c * (A @ A)
                X = a * X + B @ X
                
            if transposed:
                X = X.T
                
            X = X.astype(parameter.dtype).reshape(orig_shape)
            scale = max(1.0, orig_shape[0] / (parameter.size // orig_shape[0])) ** 0.5
            
            return parameter - lr * scale * X
            
        else:
            # ----------------- ADAMW PATH -----------------
            lr = self.get_adam_lr
            lr = lr.astype(gradient.dtype) if hasattr(lr, 'astype') else mx.array(lr, dtype=gradient.dtype)
            
            if self.weight_decay > 0.0:
                parameter = parameter - lr * self.weight_decay * parameter
                
            b1, b2 = self.betas
            step = state["step"] + 1
            state["step"] = step
            
            m = state["m"]
            v = state["v"]
            
            m = b1 * m + (1.0 - b1) * gradient
            v = b2 * v + (1.0 - b2) * mx.square(gradient)
            
            state["m"] = m
            state["v"] = v
            
            m_hat = m / (1.0 - (b1 ** step))
            v_hat = v / (1.0 - (b2 ** step))
            
            adam_update = m_hat / (mx.sqrt(v_hat) + self.eps)
            return parameter - lr * adam_update

def loss_fn(model, x, y):
    logits = model(x)
    loss = nn.losses.cross_entropy(logits, y)
    return mx.mean(loss)

def loss_fn_eval(model, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction='none')

# ---------------------------------------------------------------------------
# Execution Setup
# ---------------------------------------------------------------------------

DEVICE_BATCH_SIZE = 1
TOTAL_BATCH_SIZE = 2**14 # ~16K tokens per optimizer step for stability

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()

# Base Configuration: Optimized for stability across all Apple Silicon (160M parameters)
config = QwenConfig(
    sequence_len=MAX_SEQ_LEN, 
    vocab_size=vocab_size,
    n_layer=12, 
    n_head=8, 
    n_kv_head=2, 
    n_embd=1024
)

model = QwenHybrid(config)

def cast_to_bf16(v):
    if isinstance(v, dict):
        return {k: cast_to_bf16(x) for k, x in v.items()}
    elif isinstance(v, list):
        return [cast_to_bf16(x) for x in v]
    elif isinstance(v, mx.array) and v.dtype == mx.float32:
        return v.astype(mx.bfloat16)
    return v

model.update(cast_to_bf16(model.parameters()))
mx.eval(model.parameters())

def count_params(v):
    if isinstance(v, dict):
        return sum(count_params(x) for x in v.values())
    elif isinstance(v, list):
        return sum(count_params(x) for x in v)
    return v.size

total_params = count_params(model.parameters())
print(f"Model Architecture: Qwen3.5 Hybrid (3:1 Linear/Full)")
print(f"Total Parameters:   {total_params / 1e6:.1f}M")

optimizer = MuonAdamW(learning_rate=0.0001, muon_lr=0.02, vocab_size=vocab_size)
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

@mx.compile
def step(x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    return loss, grads

# ---------------------------------------------------------------------------
# The Autonomous Loop
# ---------------------------------------------------------------------------

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
print(f"Starting 5-minute training loop on MLX...")

grad_accum_steps = max(1, TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
print(f"Gradient accumulation steps: {grad_accum_steps}")

t_start = time.time()
step_count = 0
total_training_time = 0

def tree_map_add(t1, t2):
    if isinstance(t1, dict):
        return {k: tree_map_add(t1[k], t2[k]) for k in t1}
    elif isinstance(t1, list):
        return [tree_map_add(a, b) for a, b in zip(t1, t2)]
    return t1 + t2

def tree_map_div(t, n):
    if isinstance(t, dict):
        return {k: tree_map_div(v, n) for k, v in t.items()}
    elif isinstance(t, list):
        return [tree_map_div(v, n) for v in t]
    return t / n

while True:
    t0 = time.time()
    
    # Gradient Accumulation Loop
    accum_grads = None
    accum_loss = 0.0
    
    for _ in range(grad_accum_steps):
        x, y, epoch = next(train_loader)
        loss, grads = step(x, y)
        accum_loss += loss
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map_add(accum_grads, grads)
        
        # Prevent OOM by evaluating the accumulated graph every micro-step
        mx.eval(accum_loss, accum_grads)
            
    # Average and Update
    accum_grads = tree_map_div(accum_grads, grad_accum_steps)
    accum_loss = tree_map_div(accum_loss, grad_accum_steps)
    
    optimizer.update(model, accum_grads)
    mx.eval(model.parameters(), optimizer.state, accum_loss) 
    
    dt = time.time() - t0
    
    # Compilation Warmup: only count time after the first 5 compiled steps
    if step_count > 5:
        total_training_time += dt
    
    step_count += 1
    
    if step_count % 1 == 0:
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        print(f"\rStep {step_count:04d} | Loss: {accum_loss.item():.4f} | Progress: {progress*100:.1f}% | dt: {dt*1000:.0f}ms", end="")

    if total_training_time >= TIME_BUDGET:
        break

print("\n---")
model.eval()
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

print(f"val_bpb:          {val_bpb:.6f}")
print(f"num_steps:        {step_count}")
print(f"training_seconds: {total_training_time:.1f}")
