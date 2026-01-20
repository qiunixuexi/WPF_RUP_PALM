# Physics-Aware Learning for Detecting Robust Universal Perturbation Attacks in Wind Power Forecasting

This repository provides the official implementation of our paper **"Physics-Aware Learning for Detecting Robust Universal Perturbation Attacks in Wind Power Forecasting"**, submitted to *Engineering Applications of Artificial Intelligence (EAAI)*.

The code includes:
- 🎯 **RUP** (Robust Universal Perturbation): A weighted ensemble-based universal perturbation method targeting weather forecast inputs to degrade wind power prediction.
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
│ ├── 01_train_forecasting_models.ipynb # Train target & surrogate wind power forecasting models
│ ├── 02_generate_RUP_and_RUPW.ipynb # Generate UP, RUP (weighted), and RUPW (unweighted)
│ ├── 03_attack_performance_comparison.ipynb # Evaluate MAE and efficiency of various attack methods
│ ├── 04_train_PALM_detector.ipynb # Train the PALM detection model
│ └── 05_detection_performance_comparison.ipynb # Compare PALM vs. baseline detection methods
└── utils/
├── data_loader.py # Load data from public dataset
└── metrics.py # Evaluation metrics (e.g., MAE)


---

## 🔁 Execution Order
**The notebooks must be executed sequentially in the order specified below:**

(1) Execute **"01_train_forecasting_models.ipynb"** to train both the target and surrogate wind power forecasting models that serve as the foundation for subsequent experiments.

(2) Run **"02_generate_RUP_and_RUPW.ipynb"** to generate adversarial perturbations using the proposed RUP attack method.

(3) Execute **"03_attack_performance_comparison.ipynb"** to evaluate both the attack effectiveness (measured by MAE) and computational efficiency (execution time in milliseconds) of various attack methods, including RA, FGSM, PGD, AoA, UP, RUP, and RUPW.

(4) Run **"04_train_PALM_detector.ipynb"** to train the PALM anomaly detection model using the generated attack samples.

(5) Execute **"05_detection_performance_comparison.ipynb"** to compare the detection performance of PALM against baseline methods (Three-sigma, Boxplot, Isolation Forest, VAR, Autoencoder, and ODIN).

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
