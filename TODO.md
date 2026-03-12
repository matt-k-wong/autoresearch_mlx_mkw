# Future Research & TODOs

This document outlines the advanced architectural goals and optimizations for the `autoresearch_mlx_mkw` project. These represent the next frontier for pushing Apple Silicon to its absolute limits.

---

### 🎉 Achievements Unlocked (Deep Think Integrations)
Before attempting the tasks below, recognize the extreme optimizations already running in the baseline `train.py`:
1.  **Muon MLX Optimizer:** The repo utilizes a custom, drop-in `MuonAdamW` optimizer. It successfully routes internal 2D parameters (Attention, MLP) through a "Polar Express" Newton-Schulz 5 orthogonalization loop, keeping the math entirely within `mx.array` boundaries to ensure `@mx.compile` traces perfectly.
2.  **Chunkwise Linear Attention:** To prevent Unified Memory OOM errors on Mac laptops, the massive $T \times D^2$ cumulative sum required for DeltaNet-style linear attention is replaced with a highly vectorized, chunk-wise parallel block algorithm. This enables training at a full 2048 sequence length on a 16GB MacBook Air.
3.  **Memory-Safe Gradient Accumulation:** Implemented a robust `tree_map` accumulation loop using `mx.eval` on intermediate gradients. This allows for stable training with large batch sizes (e.g., 16k tokens) on 16GB Macs without OOM or graph explosion.
4.  **Scientific JIT Warmup Timer:** Added a warmup phase (first 5 steps) to exclude Metal kernel compilation time from the 5-minute training budget, ensuring `val_bpb` comparisons are based on actual training.
5.  **GeGLU Activation:** Successfully swapped SwiGLU for GeGLU (GELU gating). In our 5-minute benchmark, GeGLU improved `val_bpb` from **2.997** to **2.994** for a ~160M parameter model on Apple Silicon.

---

### 1. MLX Distributed (Multi-Mac Clustering)
*   **Current State:** Single-node training.
*   **The Goal:** Integrate `mlx.core.distributed` to allow the training loop to scale across 2 or more Macs connected via Thunderbolt or high-speed Ethernet.
*   **The Architecture:** Implement Data Parallelism within the `loss_and_grad_fn` step.

### 2. Dynamic Sequence Length Scaling (Curriculum)
*   **Current State:** Fixed at `MAX_SEQ_LEN = 2048`.
*   **The Goal:** Implement a curriculum learning approach where the 5-minute loop starts at 256 tokens and dynamically ramps up to 2048 tokens based on the current memory pressure (`mx.metal.get_peak_memory()`).

### 3. Mixture-of-Experts (MoE) Port
*   **Goal:** Port a Gated Mixture of Experts (like DeepSeek-V3) to MLX to see if we can train a much larger model (sparse) within the same 5-minute memory and time budget.
