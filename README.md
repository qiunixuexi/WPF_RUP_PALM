# Physics-Aware Learning for Detecting Robust Universal Perturbation Attacks in Wind Power Forecasting

This repository provides the official implementation of our paper **"Physics-Aware Learning for Detecting Robust Universal Perturbation Attacks in Wind Power Forecasting"**, submitted to *Engineering Applications of Artificial Intelligence (EAAI)*.

The code includes:
- 🎯 **RUP** (Robust Universal Perturbation): A weighted ensemble-based universal perturbation method targeting weather forecast inputs (e.g., wind speed, direction) to degrade wind power prediction.
- 🛡️ **PALM** (Physics-Aware Learning for Manipulation detection): A lightweight, physics-informed detector to identify adversarial manipulations in meteorological data.
- ⚔️ Baseline attack methods: RA, FGSM, PGD, AoA, UP,RUPW.
- 🔍 Baseline detection methods: Three-sigma rule, boxplot-based anomaly detection, Isolation Forest, Vector Autoregression (VAR), Autoencoder, and ODIN.
- 📊 Ablation studies for both RUP (via RUPW/UP) and PALM.

✅ **Full source code is now publicly available**  
✅ Built with Jupyter Notebooks for reproducibility  
✅ Uses publicly available wind power data

---

## 📦 Repository Structure
WPF_RUP_PALM/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_train_forecasting_models.ipynb          # Train target & surrogate wind power forecasting models
│   ├── 02_generate_RUP_and_RUPW.ipynb             # Generate UP, RUP (weighted), and RUPW (unweighted)
│   ├── 03_attack_performance_comparison.ipynb     # Evaluate MAE of RA, FGSM, PGD, AoA, UP, RUP, RUPW
│   ├── 04_train_PALM_detector.ipynb               # Train the PALM detection model
│   ├── 05_detection_performance_comparison.ipynb  # Compare PALM vs. three-sigma, boxplot detection
│   └── 06_PALM_ablation_and_noise_robustness.ipynb # Ablation study and noise robustness test for PALM
└── utils/
├── init.py
├── data_loader.py        # Load data from public dataset
├── metrics.py            # Evaluation metrics (e.g., MAE)
└── attack_utils.py       # Shared functions for perturbation generation


> 🔁 **Recommended execution order**: Run notebooks in numerical order (01 → 06).

---

## 🚀 Getting Started

### 1. Clone this repository
```bash
git clone https://github.com/qiunixuexi/WPF_RUP_PALM.git
cd WPF_RUP_PALM

2. Install dependencies
pip install -r requirements.txt

3. Download the dataset
The experiments use the public open-source power dataset from Texas A&M University:
🔗 https://github.com/tamu-engineering-research/Open-source-power-dataset

4. Launch Jupyter and run notebooks
jupyter notebook
Open and execute the notebooks in order (01 → 06) to reproduce the results in the paper.

🙏 Acknowledgements
We thank the Texas A&M University team for providing the open-source power dataset.
This work is submitted to EAAI and is currently under peer review.
