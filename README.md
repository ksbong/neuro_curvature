# Neuro-Geometric SNN: Phase-Space Trajectory Encoding for Motor Imagery BCI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-SNN-orange)
![Environment](https://img.shields.io/badge/Env-Google_Colab_(T4_GPU)-success)
![Status](https://img.shields.io/badge/Status-Research_Prototype-yellow)

## 🧠 Project Overview

This project introduces a novel **Neuromorphic Brain-Computer Interface (BCI)** framework that successfully bridges **Differential Geometry** and **Spiking Neural Networks (SNNs)**.

Traditional Deep Learning models (e.g., CNNs, Transformers) for EEG decoding suffer from high power consumption and require extensive user calibration time ("calibration fatigue"). To overcome this, our approach maps 1D EEG signals into a multidimensional phase-space using the Hilbert Transform, extracting highly robust geometric invariants. These features are then fed into an energy-efficient SNN, enabling **Rapid Few-Shot Calibration** for practical, real-world BCI applications.

---

## 🚀 Key Innovations

### 1. Topological Phase-Space Feature Extraction
Instead of relying on conventional amplitude or spectral power, we analyze the **geometric shape** of the brain's dynamical system.
- **Analytic Trajectory:** 1D EEG signals are mapped to a 2D complex phase plane via Hilbert Transform.
- **Curvature ($\kappa$) & Tangling:** Extracts the non-linear "twisting" of the neural trajectory. These metrics are highly robust against amplitude scaling and inter-subject skull variations.
- **[Experimental] 3D Spatiotemporal Torsion ($\tau$):** An optional/upcoming extension that maps the trajectory into a 3D space `[Real, Imaginary, Time]` to compute the true mathematical torsion (out-of-plane twisting) of the neural dynamics.

### 2. Spiking Neural Network (SNN) Decoder
- **Energy-Efficient Encoding:** Utilizes Leaky Integrate-and-Fire (LIF) neurons (`snnTorch`) with Surrogate Gradient Descent to maintain biological plausibility and ultra-low power consumption, targeted for neuromorphic chips (e.g., Intel Loihi).
- **OOM-Safe Batch Processing:** Engineered to handle massive cross-subject geometric feature extraction without VRAM overflow on standard GPUs.

### 3. Rapid Few-Shot Calibration
- Achieves highly accurate personalization using **less than 10% of target subject data** (approx. 1 minute of Motor Imagery), dramatically reducing the calibration time required for new BCI users.

---

## 🏗️ Model Architecture

*(Diagram Placeholder: Conceptual Flow from Raw EEG $\rightarrow$ Hilbert Transform $\rightarrow$ Geometric Feature Extractor (Curvature/Tangling) $\rightarrow$ Population Encoder $\rightarrow$ SNN LIF Layers $\rightarrow$ 4-Class Output)*

---

## 📊 Performance Benchmarks

Evaluated on the **Physionet EEG Motor Movement/Imagery Dataset** (4-Class MI: Left Hand, Right Hand, Both Fists, Both Feet).

| Model / Approach | Calibration Requirement | Target Accuracy (4-Class) |
| :--- | :---: | :---: |
| **Chance Level** | - | 25.00% |
| **EEGNet (CNN SoA)** | Zero-shot (Global) | 65.07% |
| **Neuro-Geo SNN (Ours)**| **Zero-shot (Global)** | **~62.90%** |
| **Neuro-Geo SNN (Ours)**| **Few-Shot Fine-tuned** | **~76.40%** |

> **Highlight:** Despite the mathematical constraints and information loss inherent to discrete spike-based communication in SNNs, our geometric features allow the model to rival floating-point CNNs in zero-shot generalization, while breaking the 76% barrier with minimal fine-tuning.

---

## 💻 Running Environment

This project has been optimized for cloud-based GPU environments to handle intensive geometric computations and SNN time-step loops.

- **Recommended Environment:** Google Colab (T4 GPU) or equivalent.
- **Memory Management:** Integrated `OOM-Safe Batch Processing` allows feature extraction of large cross-subject datasets (e.g., 20+ subjects) within a 15GB VRAM limit.
- **Frameworks:** Python, PyTorch, `snnTorch`, `mne` (for EEG preprocessing).

---

## ⚙️ Quick Start (Colab)

1. Mount Google Drive for caching the Physionet dataset.
2. Run the feature extraction block (processes data in chunks of 128 to prevent OOM).
3. Execute `Stage 1: Pre-training` to generate the global geometric backbone (`pretrained_snn.pth`).
4. Execute `Stage 2: Few-Shot Calibration` to evaluate target subject personalization.