# Future Research & TODOs

This document outlines the advanced architectural goals and optimizations for the `autoresearch_mlx_mkw` project. These represent the next frontier for pushing Apple Silicon to its absolute limits.

---

### 🎉 Achievements Unlocked (Deep Think Integrations)
Before attempting the tasks below, recognize the extreme optimizations already running in the baseline `train.py`:
1.  **Muon MLX Optimizer:** The repo utilizes a custom, drop-in `MuonAdamW` optimizer. It successfully routes internal 2D parameters (SwiGLU, Attention) through a "Polar Express" Newton-Schulz 5 orthogonalization loop, keeping the math entirely within `mx.array` boundaries to ensure `@mx.compile` traces perfectly without stalling the CPU.
2.  **Chunkwise Linear Attention:** To prevent Unified Memory OOM errors on Mac laptops, the massive $T \times D^2$ cumulative sum required for DeltaNet-style linear attention is replaced with a highly vectorized, chunk-wise parallel block algorithm. This enables training a ~686M parameter model at a full 2048 sequence length on a 16GB MacBook Air.

---

### 1. MLX Distributed (Multi-Mac Clustering)
*   **Current State:** Single-node training.
*   **The Goal:** Integrate `mlx.core.distributed` to allow the training loop to scale across 2 or more Macs connected via Thunderbolt or high-speed Ethernet.
*   **The Architecture:** We need to implement Data Parallelism (or Expert Parallelism if we move to an MoE model) within the `loss_and_grad_fn` step.

### 4. Dynamic Sequence Length Scaling
*   **Current State:** Fixed at `MAX_SEQ_LEN = 512` to prevent OOM on 64GB/128GB Macs during the Qwen-scale backward pass.
*   **The Goal:** Implement a curriculum learning approach where the 5-minute loop starts at 256 tokens and dynamically ramps up to 2048 tokens based on the current memory pressure (`mx.metal.get_peak_memory()`).

### 5. Shift from SwiGLU to GeGLU
*   **Idea:** While Qwen uses SwiGLU, some recent papers suggest GeGLU (using GELU instead of SiLU) might offer slightly better gradient flow in extremely deep models. 
*   **Agent Task:** Have the autonomous agent swap the activation function and measure the `val_bpb` impact over a 10-experiment run.
