# Apple Mac OS X MLX Autoresearch

A native Apple Silicon port of Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) project, heavily modified to run optimally on edge hardware (like an M4 MacBook Air) without melting your unified memory.

*Built with the help of Gemini 3.1 Pro (via the Gemini CLI).*

✅ **Tested On:** M4 MacBook Air (16GB RAM) | M3 Max (64GB RAM)

## Why another port?

Running Karpathy's original PyTorch code (which assumes an H100 GPU) on a Mac just doesn't work out of the box—you hit swap memory instantly and your OS crashes. To fix this and make the 5-minute autonomous research loop viable on a laptop, we rebuilt the architecture for MLX:

*   **Pure MLX:** No PyTorch overhead. Everything uses MLX's functional paradigm and Metal acceleration.
*   **Gated Hybrid Architecture:** Instead of standard toy models, we use a hybrid of **Gated DeltaNet (Linear Attention)** and standard attention (3:1 ratio). Linear attention provides $O(N)$ memory scaling, crucial for long-sequence training on laptops.
*   **Memory-Safe Gradient Accumulation:** We implemented a robust accumulation loop that uses `mx.eval` on intermediate gradients. This allows for stable training with large batch sizes (e.g., 16k tokens) while keeping peak VRAM usage under 8GB.
*   **Scientific JIT Warmup:** Metal kernel compilation can take 30-60 seconds. We've added a warmup timer that excludes the first 5 steps from the `TIME_BUDGET`, ensuring your 5-minute loop is spent on actual research, not compilation.
*   **Pure MLX Muon Optimizer:** A custom port of the Newton-Schulz 5 orthogonalization loop that traces perfectly with `@mx.compile`.

## Setup

1. **Install dependencies:**
   ```bash
   cd autoresearch_mlx_mkw
   uv sync
   ```

2. **Download the data:**
   We use the TinyStories dataset for fast feedback on local hardware.
   ```bash
   uv run prepare.py
   ```

3. **Get your baseline:**
   Run the 5-minute training script to establish your starting `val_bpb` score.
   ```bash
   uv run train.py
   ```

## Scaling for Power Users (32GB / 64GB / 128GB Macs)

If you have a high-spec Mac Studio or MacBook Pro, you can scale this experiment significantly to challenge the autonomous agent.

### 1. Scaling the Model
Modify the `config` in `train.py` to target larger parameter counts:

| Mac RAM | Targeted Model Size | `n_embd` | `n_layer` | `DEVICE_BATCH_SIZE` |
| :--- | :--- | :--- | :--- | :--- |
| **16GB** (Default) | ~160M | 1024 | 12 | 1 |
| **32GB** | ~400M | 1536 | 12 | 2 |
| **64GB** | ~0.8B | 1536 | 24 | 4 |
| **128GB** | ~1.5B | 2048 | 24 | 8 |

### 2. Increasing Stability
For larger models, you should increase the `TOTAL_BATCH_SIZE` to stabilize the gradients:
```python
# In train.py
TOTAL_BATCH_SIZE = 2**16 # 65k tokens per step for 0.8B+ models
```

### 3. Long-Context Research
Because we use **Chunkwise Linear Attention**, you can safely push the context length without OOM:
1. In `prepare.py`, change `MAX_SEQ_LEN = 4096` (or 8192).
2. Re-run `uv run prepare.py` (it's fast).
3. In `train.py`, ensure `MAX_SEQ_LEN` matches.

## Performance: Apple Silicon vs. NVIDIA DGX Spark

*   **Raw Compute:** A DGX Spark (Grace Blackwell) will process 5x to 10x more tokens in 5 minutes than an M-series Mac.
*   **The MLX Advantage:** This repository is about **Extreme Efficiency**. An M4 MacBook Air drawing ~20W can run this entire sovereign research loop silently on your lap. By using $O(N)$ linear attention and the Muon optimizer, we've closed the software gap, proving that you don't need a Petaflop desktop to do meaningful autonomous architectural research.

## Technical Trade-offs & Inductive Biases

To make this research loop viable on a laptop, we made specific architectural trade-offs that differ from the original H100/PyTorch baseline:

*   **Linear vs. Full Attention (State Saturation):** Our $O(N)$ Gated DeltaNet layers compress history into a fixed-size $D \times D$ state. While this allows for 2048+ context lengths on 16GB RAM, it can suffer from "state saturation" in complex sequences. We mitigate this with a **3:1 Hybrid Ratio**, where every 4th layer is a "Full Attention" anchor to restore precision.
*   **TinyStories vs. General Corpora:** We use the TinyStories dataset because it has lower entropy. This ensures that in a **5-minute window**, you see a meaningful drop in `val_bpb`, providing a strong signal for the autonomous agent to iterate on. A larger dataset (like `climbmix`) would be too "noisy" to show architectural improvements in such a short time.
*   **Step Count vs. Gradient Stability:** By using **Gradient Accumulation**, each "Step" in the log represents 16k tokens. You will see fewer total steps (e.g., 30-50) in 5 minutes compared to the original, but each step is mathematically more stable and less prone to the "spiky" loss curves typical of small-batch training.

## The Autonomous Loop

Once you have your baseline, point your favorite coding agent (Claude 3.7, GPT-4o, etc.) at this folder and tell it to read `program.md`. 

Check out `TODO.md` to see what we've already achieved (like the GeGLU vs SwiGLU experiments) and what you can challenge your agent with next.
