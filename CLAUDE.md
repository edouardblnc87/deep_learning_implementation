# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

This project uses `uv` for dependency management with Python 3.13+.

```bash
uv sync          # Install dependencies
uv run jupyter notebook  # Launch notebooks
uv run python main.py    # Run entry point
```

## Architecture

This is an educational deep learning framework built from scratch in NumPy, applied to S&P 500 financial time series prediction.

### Core Framework: `src/dl_utils/`

Clean abstraction stack built for pedagogical clarity:

- **Operation.py** — Atomic differentiable ops (WeightMultiply, BiasAdd, activations, Dropout). Each op implements `_output()` and `_input_grad()`.
- **Layer.py** — `Dense` layer: composes operations, supports 6 weight init strategies (standard, glorot_normal/uniform, he_normal/uniform, lecun_normal). Lazy weight initialization on first forward pass.
- **NeuralNetwork.py** — Chains layers, owns loss, exposes `train_batch()`.
- **Trainer.py** — Full training loop with shuffling, validation, patience-based early stopping, and best-model checkpointing.
- **Optimizer.py** — `SGD` and `SGD_MOMENTUM` with optional learning rate decay.
- **Loss.py** — `MeanSquaredError` and `MeanAbsoluteError`.
- **Rnn.py** — PyTorch-based RNN components (separate from the NumPy stack).

### Data Pipeline: `src/data_utils/`

- **Download_yf.py** — Downloads S&P 500 data via yfinance; `download_log_return_series()` is the preferred entry point.
- **Dataset.py** — Builds sliding-window supervised datasets. `build_dataset_abs_returns_sequential()` produces 3D tensors `(batch, seq_len, features)` for RNN input.

### Model Analysis: `src/model_analysis/`

- **Weights_initialization.py** — Captures layer signals (pre/post-activation) and plots weight/activation distributions before and after training to diagnose initialization issues.

### Legacy / Educational Reference

- **src/maths_utils.py** — Ground-up math: numerical differentiation, chain rule, manual backprop for linear regression.
- **src/neural_network_utils.py** — Pre-framework two-layer NN reference implementation.

## Notebooks

Notebooks are the primary interface; numbered in learning order:

| Notebook | Topic |
|---|---|
| `1_test.ipynb` | Math foundations (chain rule, numerical gradients) |
| `2_fnn.ipynb` | Feedforward NN on California Housing |
| `3_dl.ipynb` | **Main experiment**: dl_utils on S&P 500 returns vs. volatility prediction |
| `4_activations.ipynb` | Activation function reference |
| `5_weight_init.ipynb` | Weight initialization experiments |
| `6_rnn.ipynb` | PyTorch RNN on S&P 500 volatility |

## Key Design Decisions

- **NumPy backprop**: gradients flow through `Operation` objects via manual `_input_grad()` implementations — no autograd.
- **In-place parameter updates**: optimizers modify NumPy arrays in-place; avoid copying parameter lists.
- **Two prediction tasks**: raw returns (near-random walk, hard) vs. absolute returns/volatility (ARCH effect, tractable with NNs).
- **PyTorch used only in `Rnn.py`** — the rest of the framework is pure NumPy + scikit-learn for preprocessing.
- **No test suite** — validation is done inline in notebooks (shape assertions, comparison against PyTorch reference outputs).
