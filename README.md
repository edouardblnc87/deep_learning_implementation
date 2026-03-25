# Deep Learning from Scratch

A ground-up implementation of deep learning in pure NumPy, applied to financial time series forecasting on SP500 data. The goal is to understand every component of a neural network — forward pass, backpropagation, optimizers, regularization — by building it without relying on any deep learning framework.

---

## Project Structure

```
dl_from_scratch/
├── src/
│   ├── dl_utils/          # Core deep learning library (layers, loss, optimizer, trainer)
│   ├── data_utils/        # Financial data download and dataset construction
│   ├── model_analysis/    # Visualization utilities (weight/activation distributions)
│   ├── maths_utils.py     # Low-level math: chain rule, manual gradient, simple linear model
│   └── neural_network_utils.py  # Early-stage neural network (pre dl_utils)
├── data/                  # Cached SP500 log return series (CSV)
├── 1_test.ipynb
├── 2_fnn.ipynb
├── 3_dl.ipynb
├── 4_activations.ipynb
└── 5_weight_init.ipynb
```

---

## Notebooks

### `1_test.ipynb` — Math Foundations
Covers the mathematical building blocks of neural networks:
- Function composition and the chain rule visualized
- Manual gradient computation through composed functions (sigmoid, square)
- A first from-scratch linear model trained with gradient descent on California Housing data

### `2_fnn.ipynb` — Feedforward Neural Network (Early Implementation)
A simple feedforward neural network built directly in `neural_network_utils.py`, without the modular `dl_utils` framework. Trained on California Housing regression. Serves as a stepping stone before the full object-oriented implementation.

### `3_dl.ipynb` — Deep Learning on Financial Data
The main experiment notebook. Uses the full `dl_utils` framework to train neural networks on SP500 daily log return data.
- **Task 1**: predict next-day return (log returns) — shows that neither linear regression nor neural networks can beat a near-random baseline due to the efficient market hypothesis
- **Task 2**: predict next-day absolute return (volatility) — exploits the ARCH effect (autocorrelated volatility) where neural networks outperform linear regression
- Benchmarks: linear regression vs sigmoid network vs ReLU network vs PyTorch equivalent
- Demonstrates the effect of normalization, learning rate, early stopping, and momentum

### `4_activations.ipynb` — Activation Functions
Visual reference for all implemented activation functions.
- Each function plotted alongside its derivative (computed via the Operation's own `backward()`)
- Markdown pros/cons for each: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish
- Side-by-side overview grid of all six

### `5_weight_init.ipynb` — Weight Initialization
Experiments on how weight initialization affects training dynamics.
- For each initialization strategy: plots weight distributions, pre-activation signals, and post-activation signals before and after training
- Shows concretely how bad initialization (standard normal + Sigmoid) causes saturation while Glorot keeps signals stable
- Final MAE comparison table across all init strategies
- Strategies covered: Standard, Glorot Normal, Glorot Uniform, He Normal, He Uniform, LeCun Normal

---

## Modules

| Module | Description |
|--------|-------------|
| `dl_utils` | Core framework: operations, layers, loss functions, optimizers, trainer |
| `data_utils` | SP500 data download via yfinance, lag-feature dataset construction |
| `model_analysis` | Weight/activation distribution plotting utilities used in notebooks |
| `maths_utils` | Chain rule, manual gradients, simple linear regression from scratch |
| `neural_network_utils` | Pre-framework neural network, kept as a reference implementation |

See the `README.md` inside each module folder for detailed documentation.

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**1. Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repo and navigate into it:**
```bash
git clone <repo-url>
cd dl_from_scratch
```

**3. Create the virtual environment and install dependencies:**
```bash
uv sync
```

This reads `pyproject.toml` and installs all dependencies (numpy, matplotlib, scikit-learn, torch, yfinance) into a local `.venv`.

**4. Register the kernel for Jupyter:**
```bash
uv run python -m ipykernel install --user --name dl-from-scratch
```

**5. Launch Jupyter:**
```bash
uv run jupyter notebook
```

Then select the `dl-from-scratch` kernel when opening any notebook.

**Dependencies** (from `pyproject.toml`):

| Package | Use |
|---------|-----|
| `numpy` | Core numerical computing |
| `matplotlib` | Plotting |
| `scikit-learn` | Benchmarking, preprocessing, train/test split |
| `torch` | PyTorch reference implementations for validation |
| `yfinance` | SP500 data download |

> Python 3.13+ required.

---

## Quick Start

```python
import sys
sys.path.append('./src')
import numpy as np
from dl_utils import *
from data_utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data
df = pd.read_csv('data/df_sp_500_log_ret.csv', index_col='Date')
X, y = build_dataset_abs_returns(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test  = scaler_x.transform(X_test)

# Build model
model = NeuralNetwork(
    layers=[
        Dense(neurons=32, activation=ReLU(), weight_init='he_normal', dropout=0.8),
        Dense(neurons=16, activation=ReLU(), weight_init='he_normal', dropout=0.8),
        Dense(neurons=1,  activation=Linear())
    ],
    loss=MeanSquaredError()
)

# Train
trainer = Trainer(model, SGD_MOMENTUM(lr=0.001, final_lr=0.0001))
trainer.fit(X_train, y_train, X_test, y_test,
            epochs=300, eval_every=10, patience=5)

# Evaluate
preds = model.forward(X_test, inference=True)
print(f"MAE: {np.mean(np.abs(preds - y_test)):.4f}")
```
