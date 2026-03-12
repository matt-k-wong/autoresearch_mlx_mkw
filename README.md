# Apple Mac OS X MLX Autoresearch

A native Apple Silicon port of Andrej Karpathy's `autoresearch` project, heavily modified so it actually runs on edge hardware (like an M4 MacBook Air) without melting your unified memory.

*Built with the help of Gemini 3.1 Pro (via the Gemini CLI).*

✅ **Tested On:** M4 MacBook Air (16GB RAM)

## Why another port?

Running Karpathy's original PyTorch code (which assumes an H100 GPU) on a Mac just doesn't work out of the box—you hit swap memory instantly and your OS crashes. To fix this and make the 5-minute autonomous research loop viable on a fanless laptop, we threw out the base GPT architecture and completely rebuilt the baseline:

*   **Pure MLX:** We stripped out PyTorch. Everything uses MLX's functional paradigm and Metal acceleration, so there are zero device-transfer overheads.
*   **True Qwen3.5-0.8B Architecture:** Instead of a generic toy model, the baseline is a ~686M parameter hybrid model. It uses a 3:1 ratio of **Gated DeltaNet (Linear Attention)** to full attention. Linear attention is the secret sauce here—its $O(N)$ memory scaling lets you train a deep 24-layer model on a laptop.
*   **The Deep Think Optimizations:** We implemented a custom **Chunkwise Linear Attention** algorithm to break the $T \times D^2$ memory bottleneck, allowing the `MAX_SEQ_LEN` to run at a full **2048 tokens**. We also ported Karpathy's highly complex **Muon (Newton-Schulz 5)** optimizer directly into pure `mx.array` operations for extreme training speed.
*   **TinyStories:** We swapped the dataset. The original `climbmix` is too complex to show meaningful learning in a 5-minute window on a Mac. TinyStories lets your AI agent actually see the loss curve drop, giving it real feedback to iterate on.

## Setup

1. **Install dependencies:**
   ```bash
   cd autoresearch_mlx_mkw
   uv sync
   ```

2. **Download the data:**
   We use the `huggingface-cli` to grab the TinyStories dataset. (It's a public dataset, so you don't need to be logged into Hugging Face or provide a token).
   ```bash
   uv run prepare.py
   ```

3. **Get your baseline:**
   Run the 5-minute training script to establish your starting `val_bpb` (Validation Bits-Per-Byte) score.
   ```bash
   uv run train.py
   ```

## Scaling Up (For Mac Studio / M-Series Ultra)

If you have a more powerful machine (e.g., an M3/M4 Max or Ultra with 64GB+ of Unified Memory), you can easily scale this experiment up to push the boundaries of local AI research.

### Tweaking the Architecture
In `train.py`, you can increase the model scale by modifying the hyperparameters. For example, to target a **1.5B or 3B parameter** model:
```python
# In train.py
ASPECT_RATIO = 64
HEAD_DIM = 128
DEPTH = 36 # Increase layers
DEVICE_BATCH_SIZE = 4 # Increase batch size (if memory allows)
```
*Note: Because we use Chunkwise Linear Attention, you can also safely push `MAX_SEQ_LEN` to `4096` or `8192` in `prepare.py` and `train.py` without instantly running out of memory.*

### Swapping the Dataset
If you scale up the model, TinyStories might be too simple to challenge it. You can swap the dataset back to a larger corpus like Karpathy's original `climbmix` or `SlimPajama`.

1. In `prepare.py`, change the `BASE_URL` and the `huggingface-cli` command to point to a different dataset.
2. Update `MAX_SHARD` and the filename patterns to match your new dataset.
3. If the new dataset is massive, you may want to increase `TIME_BUDGET` from 300 seconds (5 minutes) to something longer (e.g., 3600 seconds for a 1-hour research loop) to allow the model to see enough tokens.

## Performance: Apple Silicon vs. NVIDIA DGX Spark

How does an Apple Silicon setup compare to a dedicated desktop supercomputer like the **NVIDIA DGX Spark** (128GB LPDDR5x, Grace Blackwell GB10)? 

The comparison comes down to **Specialization vs. Flexibility**:

*   **Raw Compute (The Spark Wins):** The DGX Spark features 5th-generation Tensor Cores and hardware-accelerated FP8/FP4 math via its Transformer Engine. For pure model training speed (tokens processed per second), a DGX Spark will easily process 4x to 10x more tokens in a 5-minute window than a top-tier Mac Studio.
*   **Scale and Precision:** The Spark can fine-tune dense models up to 70B parameters and serve inference for 200B models. An M-Series Ultra with 128GB of RAM can match this capacity, but Apple's lack of a dedicated dynamic precision engine means you're relying heavily on MLX's software quantization (e.g., QLoRA).
*   **The MLX Advantage (Efficiency):** Where this Apple OS X repository shines is in its **extreme efficiency**. While the DGX Spark is an Olympic weightlifter drawing 240W, an M4 MacBook Air drawing ~20W can run this entire sovereign research loop silently on your lap. By implementing $O(N)$ linear attention and a pure `mx.array` Muon optimizer, we closed the software gap, proving that you don't *need* a Petaflop desktop to do meaningful autonomous architectural research.

If you have a DGX Spark, use Karpathy's original PyTorch repository. If you have a Mac, this MLX port is arguably the most optimized sovereign research lab available for your hardware.

## The Autonomous Loop

Once you have your baseline, point your favorite coding agent (Claude 3.7, GPT-4o, etc.) at this folder and tell it to read `program.md`. 

Because the time budget is strictly 5 minutes, your agent will quickly realize it's working with a Mac, not a supercomputer cluster. It will naturally start experimenting with hyper-efficient architectural changes to squeeze out better performance. 

Check out `TODO.md` to see what we've already achieved and what you can challenge your agent with next.