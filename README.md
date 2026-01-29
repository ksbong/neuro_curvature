# 🧠 NeuroCurvature: Geometric EEG Analysis via SNN

> **"Bridging Neural Dynamics and Complex Geometry through Spiking Neural Networks"**

This project explores a novel approach to EEG (Electroencephalogram) analysis. Instead of treating EEG as simple 1D time-series data, we map signals onto the **Complex Plane** to interpret them as geometric trajectories. By extracting **Differential Geometric** features—specifically **Curvature**—we generate bio-inspired inputs for **Spiking Neural Networks (SNNs)**.

---

## 🚀 Core Methodology

### 1. Analytic Signal Mapping
We transform the raw real-valued EEG signal $x(t)$ into a complex analytic signal $z(t)$ using the **Hilbert Transform**. This allows us to observe the instantaneous phase and amplitude in a unified 2D space.
$$z(t) = x(t) + i\hat{x}(t) = A(t)e^{i\phi(t)}$$



### 2. Geometric Feature Extraction (Curvature)
To capture the rapid transitions in brain dynamics, we calculate the **Curvature** $\kappa$ of the trajectory in the complex plane. This metric encapsulates both phase shifts and amplitude modulations into a single geometric primitive.
$$\kappa(t) = \frac{|\dot{x}\ddot{y} - \dot{y}\ddot{x}|}{(\dot{x}^2 + \dot{y}^2)^{3/2}}$$

### 3. Spatio-temporal Spike Encoding
The extracted curvature serves as the primary driver for spike generation. By using **Latency Encoding** or **Delta Modulation**, we convert geometric "events" into discrete spikes, enabling high-efficiency inference on neuromorphic architectures.

---

## 📂 Project Structure

```text
NEURO_CURVATURE/
├── data/               # Data storage
│   ├── processed/      # Curvature-extracted or preprocessed data
│   └── raw/            # Raw EEG datasets (SEED, BCI, etc.)
├── notebooks/          # Research sandboxes & visualization
├── results/            # Model weights & performance logs
├── src/                # Core logic
│   ├── __init__.py
│   ├── features.py     # Curvature & geometric feature extraction
│   ├── models.py       # SNN architectures
│   ├── preprocess.py   # Signal cleaning & Hilbert transform
│   └── utils.py        # Helper functions
├── main.py             # Pipeline execution entry point
├── pyproject.toml      # Project metadata & dependencies (uv)
├── .python-version     # Python version for uv
├── .gitignore
└── README.md
```

## 🛠 Tech Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | Python 3.10+, **Rust** (for Neuromorphic Kernel Optimization) |
| **Signal Processing** | MNE-Python, SciPy, NumPy |
| **Neural Framework** | **snnTorch**, SpikingJelly, PyTorch |
| **Mathematics** | Complex Analysis, Differential Geometry, Topology |
| **Data Viz** | Matplotlib, Plotly (for 3D Trajectories) |

---

## 📈 Research Roadmap

- [ ] **Phase 1:** Implement SEED Dataset Preprocessing Pipeline.
- [ ] **Phase 2:** Develop Complex Trajectory Visualization Module (3D Phase-Space).
- [ ] **Phase 3:** Validate Curvature-based Encoding vs. Standard FFT/Power Spectral Density.
- [ ] **Phase 4:** Benchmark SNN Performance on Emotion Recognition tasks.
- [ ] **Phase 5:** Optimize high-performance SNN kernels using **Rust**.

---

## 🤝 Contribution

If you're interested in the intersection of **Differential Geometry** and **Computational Neuroscience**, feel free to contribute. Let's build an AI that truly understands the "geometry" of thought.