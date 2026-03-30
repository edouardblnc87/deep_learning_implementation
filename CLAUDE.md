# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A deep learning framework built from scratch in pure NumPy, applied to S&P 500 financial time series forecasting. The goal is to implement every component (forward pass, backpropagation, optimizers, regularization) without relying on deep learning frameworks. PyTorch is used only for reference/validation implementations.

## Environment Setup

Uses `uv` (Rust-based Python package manager), Python 3.13.

```bash
uv sync                                                    # install dependencies
uv run python -m ipykernel install --user --name dl-from-scratch  # register Jupyter kernel
uv run jupyter notebook                                    # launch notebooks
```

## Architecture

### Abstraction Stack (bottom to top)

```
Operation           — base unit: forward() + backward(), validates gradient shapes
  └── ParamOperation — adds learnable param + param_grad (for optimizer)
        ├── WeightMultiply (W @ X)
        └── BiasAdd (+ B)

Layer               — composes a sequence of Operations
  └── Dense         — WeightMultiply → BiasAdd → Activation → Dropout

NeuralNetwork       — stack of Layers + a Loss function

Trainer             — training loop: batching, early stopping, LR decay
```

**Key pattern — lazy initialization**: Layers create weights on the first forward pass using the actual input shape, so input dimensions don't need to be specified when constructing layers.

**Key pattern — in-place param updates**: Optimizers must use `param[:] -= lr * grad` (not `param = ...`) to mutate the shared array rather than rebind the local variable.

### Module Map

| Path | Role |
|------|------|
| `src/dl_utils/` | Core framework (Operation, Layer, NeuralNetwork, Loss, Optimizer, Trainer, Metrics) |
| `src/dl_utils/Rnn.py` | PyTorch-based wrappers for RNN experiments (reference only) |
| `src/data_utils/Download_yf.py` | Download S&P 500 log-return series via yfinance |
| `src/data_utils/Dataset.py` | Build sliding-window datasets from return series; scaling helpers |
| `src/model_analysis/` | Visualization of weight/activation distributions during training |
| `src/maths_utils.py` | Math foundations (chain rule demos, manual gradient computation) |
| `data/df_sp_500_log_ret.csv` | Cached S&P 500 log returns (used by notebooks to avoid re-downloading) |

### Notebooks

| Notebook | Focus |
|----------|-------|
| `1_test.ipynb` | Math foundations (chain rule, manual gradients) |
| `2_fnn.ipynb` | Early feedforward NN before the modular framework |
| `3_dl.ipynb` | Main application: SP500 log-return & volatility prediction |
| `4_activations.ipynb` | Activation function analysis & derivatives |
| `5_weight_init.ipynb` | Weight initialization strategies and their impact |
| `6_rnn.ipynb` | RNN experiments on financial data |

Notebooks import via `sys.path.append('./src')` and then `from dl_utils import *`.

## Key Implementation Details

- **Dropout**: Behaves differently during training vs. inference — pass `inference=True` to `model.forward()` for evaluation.
- **Weight init options**: `'standard'` (N(0,1)) or `'he'` / `'he_normal'` (scaled for ReLU).
- **Early stopping**: `Trainer` checkpoints the best model and restores it if validation loss plateaus for `patience` evaluation checks.
- **LR decay**: Linear or exponential decay from `lr` to `final_lr` over the training run.
- **No batch normalization** in the NumPy framework; available only in the PyTorch wrapper (`Rnn.py`).
