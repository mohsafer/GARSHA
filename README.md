# GARSHA

**Global Anomaly Representation via Scalable Heterogeneous Attention for Large-Scale Multivariate Time Series**

GARSHA is a research-oriented framework for **multivariate time-series anomaly detection**. The model learns heterogeneous temporal representations through multi-scale patch processing and attention mechanisms, then uses divergence-based discrepancy signals to identify anomalous behavior.

> **Status:** Research code / work in progress. Experimental configurations and interfaces may change as the project evolves.

## Overview

Large-scale multivariate time series often contain complex dependencies across variables and temporal resolutions. GARSHA is designed to address this challenge by combining:

- **Multi-scale patch representations** for modeling temporal patterns at different resolutions.
- **Global and local patch attention** for capturing long-range and fine-grained dependencies.
- **Transformer-style encoder layers** for scalable sequence modeling.
- **RevIN normalization** to improve robustness to distribution shifts across windows.
- **Divergence-based anomaly scoring**, with support for Tsallis, Rényi, and KL divergence variants.
- **Reconstruction and attention outputs** for analysis and visualization.

The implementation also includes utilities for training, evaluation, GPU configuration, parameter validation, and bifurcation analysis of model behavior under different hyperparameter settings.

## Method

Given an input window \(X \in \mathbb{R}^{L \times M}\), where \(L\) is the window length and \(M\) is the number of variables, GARSHA:

1. Normalizes the input using reversible instance normalization.
2. Decomposes the temporal window into patches at one or more scales.
3. Embeds patch contents and patch positions into a shared representation space.
4. Applies heterogeneous attention to model global and local temporal relationships.
5. Aggregates attention discrepancies across encoder layers and patch scales.
6. Produces anomaly scores for identifying unusual temporal behavior.

The model supports configurable window sizes, patch sizes, attention heads, encoder depth, embedding dimensions, divergence functions, and anomaly ratios.

## Repository Structure

```text
GARSHA/
├── main.py                    # Training and evaluation entry point
├── BifurcationAnalysis.py     # Hyperparameter and stability analysis
├── model/
│   ├── Garsha.py              # GARSHA anomaly detector
│   ├── attn.py                # Heterogeneous attention modules
│   ├── embed.py               # Input and patch embeddings
│   ├── entropy.py             # Divergence / entropy utilities
│   └── RevIN.py               # Reversible instance normalization
├── data_factory/
│   └── data_loader.py         # Dataset loading and preprocessing
├── train_factory/
│   └── trainer.py             # Training and evaluation logic
├── dataset/                   # Place datasets here when applicable
├── checkpoints/               # Saved model checkpoints
├── result/                    # Logs and evaluation outputs
└── LICENSE
```

## Installation

Create an isolated Python environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm einops
```

The recommended PyTorch installation may depend on your operating system and CUDA version. Please install the appropriate build from the official PyTorch distribution channel when GPU acceleration is required.

## Usage

### Training

The main entry point is `main.py`. A typical training command is:

```bash
python main.py \
  --mode train \
  --dataset credit \
  --data_path ./dataset/creditcard_ts.csv \
  --win_size 100 \
  --patch_size 5 \
  --input_c 9 \
  --output_c 9 \
  --num_epochs 10 \
  --batch_size 128 \
  --lr 1e-4 \
  --divergence tsallis \
  --q_param 0.5
```

### Evaluation

Use the same configuration and switch the execution mode to `test`:

```bash
python main.py \
  --mode test \
  --dataset credit \
  --data_path ./dataset/creditcard_ts.csv \
  --win_size 100 \
  --patch_size 5 \
  --input_c 9 \
  --output_c 9 \
  --divergence tsallis \
  --q_param 0.5
```

Evaluation reports accuracy, precision, recall, and F-score when the configured dataset and labels support these metrics. Logs are written under `result/`, while checkpoints are written under `checkpoints/` by default.

### Multi-scale configurations

Patch sizes must be positive divisors of the selected window size. For example, a window size of `99` can be paired with patch sizes such as `[3, 9, 33]`:

```bash
python main.py \
  --mode train \
  --win_size 99 \
  --patch_size 3 9 33 \
  --divergence kl
```

Depending on the argument parser and shell environment, multiple patch sizes may also need to be supplied through a custom configuration wrapper or adapted parser. See `main.py` for the complete set of supported options and defaults.

## Configuration Highlights

| Argument | Description | Default |
| --- | --- | ---: |
| `--win_size` | Input sequence length | `100` |
| `--patch_size` | Temporal patch size(s) | `[5]` |
| `--n_heads` | Number of attention heads | `1` |
| `--e_layers` | Number of encoder layers | `3` |
| `--d_model` | Representation dimension | `256` |
| `--lr` | Learning rate | `1e-4` |
| `--num_epochs` | Number of training epochs | `10` |
| `--batch_size` | Training batch size | `128` |
| `--divergence` | `tsallis`, `renyi`, or `kl` | `tsallis` |
| `--q_param` | Parameter for Tsallis/Rényi divergence | `0.5` |
| `--anormly_ratio` | Expected anomaly ratio in percent | `4.0` |
| `--use_gpu` | Enable GPU execution when available | `True` |

For reproducible comparisons, record the dataset split, preprocessing procedure, random seed, hardware, and all model hyperparameters alongside each experiment.

## Bifurcation and Stability Analysis

`BifurcationAnalysis.py` provides an experimental workflow for studying how training dynamics and anomaly scores change as a model parameter varies. It can compare divergence choices using:

- Final training loss
- Convergence rate
- Loss variance as a stability measure
- Anomaly-score variability
- Phase portraits and publication-oriented plots
- LaTeX and CSV summary tables

Run the analysis with:

```bash
python BifurcationAnalysis.py
```

The script can export PNG, PDF, EPS, CSV, and LaTeX artifacts for further inspection or inclusion in academic reports.

## Datasets

The training entry point contains support for dataset configurations including credit-card time series and experimental UCR / SMD-style layouts. Dataset files are not included in this repository. Before running an experiment:

1. Obtain the dataset from its official source.
2. Follow the applicable dataset license and usage terms.
3. Convert the data to the format expected by `data_factory/data_loader.py`.
4. Update `--dataset`, `--data_path`, `--input_c`, and `--output_c`.
5. Document preprocessing and anomaly-label conventions for reproducibility.

## Related Work

The implementation is informed by the following lines of research:

1. **Anomaly Transformer** — Xu et al., *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy*.
2. **DCdetector** — Yang et al., *DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection*.
3. **Stochastic recurrent anomaly detection** — Su et al., *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network*.
4. **RevIN** — Kim et al., *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*.

Please consult the original publications for the formal methods, datasets, baselines, and evaluation protocols. This repository should not be interpreted as an official implementation of those works.

## Reproducibility Checklist

When reporting results, we recommend recording:

- Dataset name, version, split, and preprocessing steps.
- Window length and patch-size configuration.
- Divergence type and divergence parameters.
- Optimizer, learning rate, batch size, and number of epochs.
- Random seed and hardware details.
- Checkpoint selection criteria.
- Anomaly-thresholding or anomaly-ratio procedure.
- Point-adjustment and evaluation conventions, if used.

## Citation

If GARSHA contributes to your research, please cite this repository and the associated paper when a formal publication becomes available:

```bibtex
@software{garsha,
  author  = {Mohsafer},
  title   = {GARSHA: Global Anomaly Representation via Scalable Heterogeneous Attention for Large-Scale Multivariate Time Series},
  year    = {2026},
  url     = {https://github.com/mohsafer/GARSHA}
}
```

## License

This project is distributed under the license included in [`LICENSE`](LICENSE).

## Contact

For questions, reproducibility issues, or research collaboration, please open a GitHub issue in this repository with:

- The command used to run the experiment
- Relevant configuration values
- Dataset and preprocessing details
- Full error output or logs
- Operating system, Python, PyTorch, and CUDA versions
