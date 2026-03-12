# Apple Mac OS X MLX Autoresearch

This is an autonomous research loop optimized for Apple Silicon and MLX.

## Goal
Optimize `train.py` to achieve the lowest possible `val_bpb` in a fixed 5-minute training window on this Mac.

## Instructions
1. **Model & Architecture**: You can modify anything in `train.py` including `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, and the `GPT` class logic.
2. **Optimizer**: Experiment with `learning_rate` or implement custom optimizers (e.g., SOAP or Muon ports to MLX).
3. **Execution**: Run `uv run train.py`. 
4. **Metric**: Extract `val_bpb` from the output.
5. **Persistence**: If `val_bpb` improves, commit the change. If not, revert.

## MLX Specifics
- Use `mx.core` (mx) and `mlx.nn` (nn).
- Ensure any functions you add to the training step are wrapped in `@mx.compile` for Metal acceleration.
- Take advantage of Unified Memory; don't bother with `.to()` calls.

## Start
Establish the baseline by running `uv run train.py` as-is.
