# Differential-Privacy Project

This repository implements privacy-preserving machine learning experiments using **differential privacy** on the **UCI Adult dataset**.  
It combines standard classifiers with **output perturbation via PyDP** to evaluate tradeoffs between privacy, utility, fairness, and membership-inference resistance.

---

## 🔧 Setup & Requirements

- Python 3.9+  
- Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended: run in a **virtual environment** or inside **Docker** (Dockerfile included).

---

## How to Run

```bash
docker build pydp-env .
```

```bash
docker run --rm -v pydp-env
```

**OR**

```bash
python main.py
```


This will:

1. Download and preprocess the dataset (if not present).  
2. Train a base classifier (RandomForest).  
3. Apply output perturbation using PyDP for a range of ε values.  
4. Evaluate and log:
   - Utility (accuracy)
   - Fairness (demographic parity)
   - Membership-Inference Attack (MIA) success  
5. Produce plots of privacy vs utility / MIA tradeoffs (saved under `./outputs`).

---

## 📊 What We Measure

- **Accuracy:** Baseline vs privatized classifier  
- **Privacy–Utility Trade-off:** Effect of different ε (privacy budgets) on accuracy  
- **Fairness:** Demographic parity over sensitive attributes (e.g., race or sex)  
- **Membership Inference Attack (MIA) success:** Baseline vs privatized model  

---

##  Experiments / Parameters

- Tested ε values: `0.1`, `0.5`, `1.0`, `2.0`, `5.0`, `10.0`  
- Random seed: `42` (for reproducibility)  
- You can change parameters in `config.py` or directly in `main.py`.

---

## 📈 Example Output

Running with ε = 0.1 … 10.0 yields results showing:

- Accuracy degradation relative to the non-private baseline.  
- Reduction in MIA success, indicating stronger privacy.  
- Variation in fairness metrics (demographic parity).  

Results (as CSV and plots) are saved under:

./outputs/results.csv
./outputs/results.png
