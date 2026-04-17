# Real-time Instantaneous Phase Estimation from Gearbox Signals Using A Deep Network With Walsh-Hadamard Transform

**Yingyi Luo, Emadeldeen Hamdan, Xin Zhu, Yue Yu, Hamid Reza Karimi, Ahmet Enis Cetin**

*Department of Electrical and Computer Engineering, University of Illinois Chicago, Chicago, IL, USA*  
*Department of Mechanical Engineering, Politecnico di Milano, Milan, Italy*

---

## Overview

Real-time estimation of the Instantaneous Phase (IPhase) from gearbox vibration signals is crucial for detecting sudden changes in mechanical systems, as variations in IPhase often indicate emerging faults. This repository contains the official implementation of a two-branch deep learning framework that incorporates a **Walsh-Hadamard Transform (WHT)** layer to estimate the instantaneous phase of the analytic signal directly from gearbox signals in real time.

By integrating transform-domain techniques into the neural architecture, the WHT method enhances the model's ability to identify critical signal patterns, resulting in more accurate and robust vibration IPhase estimation.

## Highlights

- Real-time instantaneous phase estimation from gearbox signals enables early detection of mechanical faults.
- A novel two-branch deep learning model integrates Walsh-Hadamard Transform (WHT) layers for enhanced feature extraction.
- Experimental results demonstrate improved accuracy and robustness in vibration phase estimation using the WHT-based approach.

---

## Method

### Walsh-Hadamard Transform (WHT) Layer

The WHT is a linear transformation used in signal processing and data compression, characterized by its computational efficiency. Unlike the DFT, the Hadamard matrix consists only of ±1 elements, avoiding complex exponential terms. The WHT layer consists of:

1. A forward Walsh-Hadamard transform (FWHT)
2. Learnable spectral gating and channel mixing via pointwise convolution
3. Soft- or hard-thresholding for sparsity promotion
4. An optional residual skip

### Model Variants

We propose three model variants, illustrated below:

| Model | Description | Parameters |
|-------|-------------|------------|
| **TPNWHT** | Two-Pronged Network with WHT layers (two separate real/imaginary branches) | 1,223,780 |
| **TPNWHT2** | TPNWHT with learnable downsampling (decimation) layers | 611,890 |
| **JWHT** | Joint (single-channel) WHT model with learnable downsampling | 350,286 |

**(a) TPNWHT:** Two parallel branches (1024 → 512 → 128 → 64 → WHT → 128 → 32 → WHT → 16) estimate the real and imaginary parts of the analytic signal separately.

**(b) TPNWHT2:** Adds a learnable low-pass filter + decimation stage before the linear layers, reducing the input from 1024 to 256 per branch (concatenated to 512) with anti-aliasing.

**(c) JWHT:** A single-branch joint model that trains the real and imaginary parts together (output: first 16 dims = real, last 16 dims = imaginary), targeting FPGA-friendly deployment.

---

## Results

### Synthetic Data

| Metrics | ECHT | MLP | TPNWHT | TPNWHT2 | JWHT | JDC |
|---------|------|-----|--------|---------|------|-----|
| RMSEP25S (°) | 104.05 | 1.48 | 0.45 | 0.85 | **0.0759** | 0.36 |
| MACE (°) | 29.90 | 1.15 | 0.37 | 0.09 | **0.06** | 0.26 |
| CVar | 0.46 | 1.28×10⁻⁴ | 1.24×10⁻⁵ | 1.08×10⁻⁴ | **3.4×10⁻⁷** | 8.20×10⁻⁶ |
| CSD (°) | 59.64 | 0.92 | 0.29 | 0.57 | **0.05** | 0.23 |
| NAACC (%) | 83.39 | 99.36 | 99.79 | 99.65 | **99.97** | 99.85 |

*ECHT = Empirical Complex Hilbert Transform (baseline). Lower RMSEP25S, MACE, CVar, CSD and higher NAACC are better.*

### Evaluation Metrics

- **MACE** (Mean Absolute Circular Error): average angular deviation between predicted and true phase.
- **CVar** (Circular Variance): bounded dispersion measure for angular data (0 = perfect, 1 = uniform).
- **CSD** (Circular Standard Deviation): circular analogue of standard deviation.
- **NAACC** (Normalized Angular Accuracy): percentage of predictions within an acceptable angular tolerance.
- **RMSEP25S** (Average RMSE per 0.25 s): evaluates accuracy in the time domain over 0.25-second windows.

---

## Repository Structure

```
.
├── TPNWHT/
│   ├── model_POSE_HTF.py            # TPNWHT model definition
│   ├── Train_POSE_HT_REAL.py        # Training script (real branch)
│   ├── Train_POSE_HT_IMG.py         # Training script (imaginary branch)
│   ├── test_POSE_HT_ALL-PAPER-FIGURE.py  # Testing and figure generation
│   ├── HT_result_to_excel.py        # Export results to Excel
│   ├── best_test_model_real_parts.pth    # Best model checkpoint (real)
│   ├── best_test_model_imag_parts.pth    # Best model checkpoint (imaginary)
│   └── x_train.npy / y_train.npy / x_test.npy / y_test.npy  # Data splits
│
└── TPNWHT2/
    ├── model_POSE_HTF.py            # TPNWHT2 model definition (with decimation)
    ├── Train_POSE_HT_REAL.py        # Training script (real branch)
    ├── Train_POSE_HT_IMG.py         # Training script (imaginary branch)
    ├── test_POSE_HTF.py             # Testing script
    └── best_test_model_*.pth        # Model checkpoints
```

---

## Installation

```bash
conda create -n wht_phase python=3.9
conda activate wht_phase
pip install torch numpy scipy pandas openpyxl matplotlib
```

> Requires PyTorch ≥ 1.12 and a CUDA-capable GPU (recommended).

---

## Usage

### Training

```bash
# Train TPNWHT real branch
cd TPNWHT
python Train_POSE_HT_REAL.py

# Train TPNWHT imaginary branch
python Train_POSE_HT_IMG.py
```

```bash
# Train TPNWHT2 (with decimation layer)
cd TPNWHT2
python Train_POSE_HT_REAL.py
python Train_POSE_HT_IMG.py
```

### Testing & Evaluation

```bash
# Generate evaluation figures (paper results)
cd TPNWHT
python test_POSE_HT_ALL-PAPER-FIGURE.py

# Export numerical results to Excel
python HT_result_to_excel.py
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{luo2025realtimephase,
  title     = {Real-time Instantaneous Phase Estimation from Gearbox Signals Using A Deep Network With Walsh-Hadamard Transform},
  author    = {Luo, Yingyi and Hamdan, Emadeldeen and Zhu, Xin and Yu, Yue and Karimi, Hamid Reza and Cetin, Ahmet Enis},
  journal   = {Mechanical Systems and Signal Processing},
  year      = {2025},
  publisher = {Elsevier}
}
```

---

## Contact

- **Yingyi Luo** — yluo52@uic.edu
