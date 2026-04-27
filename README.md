# Real-time Instantaneous Phase Estimation from Gearbox Signals Using A Deep Network With Walsh-Hadamard Transform

**Yingyi Luo, Emadeldeen Hamdan, Yue Yu, Xin Zhu, Hamid Reza Karimi, Zhaoye Qin, Fulei Chu, Ahmet Enis Cetin**

<sup>a</sup> *Department of Electrical and Computer Engineering, University of Illinois Chicago, Chicago, IL, USA*  
<sup>b</sup> *Department of Mechanical Engineering, Politecnico di Milano, Milan, Italy*  
<sup>c</sup> *State Key Laboratory of Tribology, Department of Mechanical Engineering, Tsinghua University, Beijing, China*

📄 [Paper](#citation) | 💻 [Code](https://github.com/Cetin-Lab/-Real_time_-Gearbox_Phase)

---

## Overview

Real-time estimation of the Instantaneous Phase (IPhase) from gearbox vibration signals is crucial for detecting sudden changes in mechanical systems — variations in IPhase often indicate emerging faults. This repository contains the official implementation of a **two-branch deep learning framework** that incorporates a **Walsh-Hadamard Transform (WHT)**-based network layer to estimate the instantaneous phase of the analytic signal directly from gearbox signals in real time.

The network constructs a latent representation in the WHT domain, enabling efficient extraction of key signal features. By integrating transform-domain techniques into the neural architecture, the WHT method enhances the model's ability to identify critical patterns, resulting in more accurate and robust vibration IPhase estimation.

## Highlights

- Real-time instantaneous phase estimation from gearbox signals enables early detection of mechanical faults.
- A novel two-branch deep learning model integrates Walsh-Hadamard Transform (WHT) layers for enhanced feature extraction.
- Experimental results demonstrate improved accuracy and robustness in vibration phase estimation using the WHT-based approach.
- Validated on both synthetic data and the real-world **Safran aircraft engine vibration dataset** (Surveillance 8 challenge).

---

## Method

### Walsh-Hadamard Transform (WHT) Layer

The WHT is a linear transformation used in signal processing and data compression, characterized by its computational efficiency. Unlike the DFT, the Hadamard matrix consists only of ±1 elements, avoiding complex exponential terms. Given a vector of length N = 2<sup>M</sup>, the Hadamard Transform is:

$$\mathbf{X} = \mathcal{H}(\mathbf{x}) = \sqrt{\frac{1}{N}} \mathbf{H}_N \mathbf{x}$$

The WHT layer consists of:
1. A forward Walsh-Hadamard transform (FWHT) with zero-padding to the nearest power of 2
2. Learnable spectral gating and channel mixing via pointwise 1×1 convolution
3. Soft- or hard-thresholding for structured sparsity promotion
4. An optional residual skip connection

### Model Variants

| Model | Description | Parameters |
|-------|-------------|------------|
| **TPNWHT** | Two-Pronged Network with WHT layers — dual branches estimate real and imaginary parts separately | 1,223,780 |
| **TPNWHT2** | TPNWHT with learnable anti-aliasing low-pass filter + decimation layers | 611,890 |
| **JWHT** | Joint (single-channel) WHT model with learnable downsampling, real and imaginary trained together | 350,286 |

**Network architectures (Figure 2):**

- **(a) TPNWHT:** Two parallel branches `1024 → 512 → 128 → 64 → WHT Layer → 128 → 32 → WHT Layer → 16` estimate Re(Y) and Im(Y) independently, then compute IPhase.
- **(b) TPNWHT2:** Adds a learnable LPF + downsampling stage per branch (`1024 → 512 → 64`) before the WHT layers, then proceeds identically to TPNWHT.
- **(c) JWHT:** Single branch `1024 → LPF+Downsample → 512 → 64 → WHT Layer → 32 → WHT Layer → 32`, outputting a 32-dimensional vector (dim 1:16 = real, dim 17:32 = imaginary) with only 350,286 parameters — suitable for FPGA deployment.

---

## Results

### Synthetic Data (Table 2)

| Metrics | ECHT | MLP | TPNWHT | TPNWHT2 | JWHT |
|---------|------|-----|--------|---------|------|
| RMSEP25S (°) | 104.05 | 1.48 | 0.45 | 0.85 | **0.0759** |
| MACE (°) | 29.90 | 1.15 | 0.37 | 0.09 | **0.06** |
| CVar | 0.46 | 1.28×10⁻⁴ | 1.24×10⁻⁵ | 1.08×10⁻⁴ | **3.4×10⁻⁷** |
| CSD (°) | 59.64 | 0.92 | 0.29 | 0.57 | **0.05** |
| Parameters | — | 1,366,592 | 1,223,780 | 611,890 | **350,286** |
| NAACC (%) | 83.39 | 99.36 | 99.79 | 99.65 | **99.97** |

*ECHT = Empirical Complex Hilbert Transform (DFT-based baseline). Bold = best.*

### Safran Aircraft Engine Vibration Dataset (Table 3)

Evaluated on the **Surveillance 8 challenge** dataset — high-resolution vibration signals from a civil aircraft engine under nonstationary operating conditions (50 kHz sampling rate, 200 s duration). Input window: 1024 samples, stride: 16.

| Model | RMSEP25S (°) | MACE (°) | CVar | CSD (°) | NAACC (%) |
|-------|-------------|---------|------|---------|----------|
| **TPNWHT** | **39.971** | **17.969** | **0.149** | **32.599** | **90.017** |
| MLP | 41.470 | 18.129 | 0.153 | 33.014 | 89.928 |
| TPNWHT2 w/ decimation | 41.373 | 18.288 | 0.162 | 34.094 | 89.840 |
| MLP w/ decimation | 41.893 | 18.727 | 0.165 | 34.436 | 89.596 |
| JWHT | 43.787 | 19.574 | 0.176 | 35.616 | 89.126 |
| JMLP | 44.424 | 21.508 | 0.213 | 39.689 | 88.051 |

### Fault-Order Estimation (Table 4)

Localization of the spectral peak closest to the theoretical fault order 23.277 via angular resampling of the Safran dataset:

| Model | Estimated order k<sub>est</sub> | Relative error ε (%) |
|-------|-------------------------------|----------------------|
| TPNWHT | 23.10721 | 0.7294 |
| TPNWHT2 | 23.41438 | 0.5902 |
| **JWHT** | **23.29200** | **0.0645** |

### Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **MACE** | Mean Absolute Circular Error — average angular deviation between predicted and true phase |
| **CVar** | Circular Variance — bounded dispersion (0 = perfect, 1 = uniform) |
| **CSD** | Circular Standard Deviation — scale-invariant angular spread |
| **NAACC** | Normalized Angular Accuracy — fraction of predictions within acceptable angular tolerance, normalized to [0,1] |
| **RMSEP25S** | Average RMSE per 0.25 s window — evaluates time-domain accuracy over sliding segments |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── models/
│   ├── __init__.py          # Exports TPNWHT, TPNWHT2, JWHT
│   ├── wht_layers.py        # Shared primitives: WHT1D, DownsampleLY, SoftThresholding
│   ├── tpnwht.py            # TPNWHT model (dual-branch, no decimation)
│   ├── tpnwht2.py           # TPNWHT2 model (dual-branch with learnable decimation)
│   └── jwht.py              # JWHT model (joint single model)
├── train.py                 # Unified training entry point (all three models)
├── test_TPNWHT.ipynb        # Evaluation notebook — TPNWHT
├── test_TPNWHT2.ipynb       # Evaluation notebook — TPNWHT2
├── test_JWHT.ipynb          # Evaluation notebook — JWHT (includes fault-order estimation)
└── checkpoints/
    ├── TPNWHT_best_test_model_real_parts.pth
    ├── TPNWHT_best_test_model_imag_parts.pth
    ├── TPNWHT2_best_test_model_real_parts.pth
    ├── TPNWHT2_best_test_model_imag_parts.pth
    └── JWHT_best_model.pth
```

Data files (`x_train.npy`, `y_train.npy`, `x_test.npy`, `y_test.npy`) should be placed in the project root directory. Expected shapes: `x_*.npy` → `(N, 1024)`, `y_*.npy` → `(N, 32)` where `y[:, :16]` = real part and `y[:, 16:]` = imaginary part.

---

## Installation

```bash
conda create -n wht_phase python=3.9
conda activate wht_phase
pip install -r requirements.txt
```

> **Requirements:** PyTorch ≥ 1.12, CUDA-capable GPU (recommended). Models were trained with the AdamW optimizer and MSE loss.

---

## Usage

### Training

All three models share a single entry point:

```bash
# TPNWHT — train real and imaginary branches separately
python train.py --model tpnwht --part real --epochs 2000
python train.py --model tpnwht --part imag --epochs 500

# TPNWHT2 — with learnable decimation layers
python train.py --model tpnwht2 --part real --epochs 2000
python train.py --model tpnwht2 --part imag --epochs 500

# JWHT — joint real+imaginary, single model
python train.py --model jwht --epochs 500
```

Checkpoints are saved automatically to `checkpoints/` with model-prefixed filenames. Training resumes from the best existing checkpoint if one is found.

### Evaluation

Open the corresponding notebook in Jupyter:

| Notebook | Model | Dataset |
|----------|-------|---------|
| `test_TPNWHT.ipynb` | TPNWHT | Synthetic / Safran |
| `test_TPNWHT2.ipynb` | TPNWHT2 | Synthetic / Safran |
| `test_JWHT.ipynb` | JWHT | Safran (includes fault-order estimation) |

Each notebook loads the checkpoint from `checkpoints/`, runs inference, prints all five metrics (MACE, CVar, CSD, NAACC, RMSEP25S), and generates the phase comparison plot and rose diagrams.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{luo2026realtimephase,
  title     = {Real-time Instantaneous Phase Estimation from Gearbox Signals Using A Deep Network With Walsh-Hadamard Transform},
  author    = {Luo, Yingyi and Hamdan, Emadeldeen and Yu, Yue and Zhu, Xin and Karimi, Hamid Reza and Qin, Zhaoye and Chu, Fulei and Cetin, Ahmet Enis},
  journal   = {Advanced Engineering Informatics},
  year      = {2026},
  publisher = {Elsevier}
}
```

---

## Acknowledgments

This work was supported by the National Science Foundation (NSF) under Grant IDEAL 2217023 and Grant POSE 2303700.

---

## Contact

- **Yingyi Luo** — yluo52@uic.edu
- **Ahmet Enis Cetin** (Corresponding) — aecyy@uic.edu
- **Hamid Reza Karimi** (Corresponding) — hamidreza.karimi@polimi.it
