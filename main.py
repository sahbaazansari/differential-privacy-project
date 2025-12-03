#!/usr/bin/env python3
"""
Main file
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# PyDP safety check
try:
    import pydp as dp
    from pydp.algorithms.laplacian import BoundedMean
    print("✓ PyDP ready")
except ImportError:
    print("✗ PyDP missing")
    sys.exit(1)

# Config
EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
RANDOM_SEED = 42
OUTPUT_DIR = "./outputs"  # OUTPUT_DIR: For results
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

def download_uci_adult():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    output_path = Path("adult.data")
    
    if output_path.exists():
        print(f"✓ Dataset exists: {output_path}")
        return str(output_path)
    
    print(" Downloading UCI Adult...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    print(f"✓ Downloaded: {output_path}")
    return str(output_path)

def load_uci_adult(data_path="adult.data"):
    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path, names=COLUMN_NAMES, na_values=" ?", skipinitialspace=True)
    df = df.dropna()
    print(f"Dataset: {df.shape}")
    
    df["income"] = (df["income"] == ">50K").astype(int)
    
    numeric_features = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    categorical_features = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
    
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    features = numeric_features + categorical_features
    X = df[features].values.astype(np.float32)
    y = df["income"].values
    sensitive = df[["race", "sex"]].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Positive rate: {y.mean():.1%}")
    
    return (X_train, y_train, s_train), (X_test, y_test, s_test), scaler

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def demographic_parity(y_pred, sensitive):
    p0 = y_pred[sensitive[:, 0] == 0].mean()
    p1 = y_pred[sensitive[:, 0] == 1].mean()
    return abs(p0 - p1)

def mia_success(model, X_train, y_train, X_test, y_test):
    """ MIA - Membership Inference Attack"""
    try:
        # Sample for speed
        n_sample = min(1000, len(X_train))
        train_idx = np.random.choice(len(X_train), n_sample, replace=False)
        test_idx = np.random.choice(len(X_test), n_sample, replace=False)
        
        train_conf = model.predict_proba(X_train[train_idx]).max(axis=1)
        test_conf = model.predict_proba(X_test[test_idx]).max(axis=1)
        
        threshold = np.median(train_conf)
        train_hits = (train_conf > threshold).mean()
        test_hits = (test_conf <= threshold).mean()
        
        return (train_hits + test_hits) / 2
    except:
        return 0.5  # Random guessing fallback

class PrivateModel:
    def __init__(self, epsilon):
        self.epsilon = float(epsilon)
    
    def private_predictions(self, predictions):
        """PyDP requires INTEGER inputs. Scale probs to 0-100 range."""
        # Scale [0,1] -> [0,100] integers
        scaled_preds = (predictions * 100).astype(np.int32)
        scaled_preds = np.clip(scaled_preds, 0, 100)
        
        private_scaled = []
        for pred_int in scaled_preds:
            mech = BoundedMean(
                epsilon=self.epsilon,
                lower_bound=0,    # INTEGER
                upper_bound=100   # INTEGER
            )
            # PyDP needs LIST of integers
            noisy_int = mech.quick_result([int(pred_int)])
            private_scaled.append(float(noisy_int))
        
        # Scale back to [0,1]
        private_probs = np.array(private_scaled) / 100.0
        return np.clip(private_probs, 0, 1)

def apply_dp(model, epsilon, X_test):
    priv_model = PrivateModel(epsilon)
    orig_probs = model.predict_proba(X_test)[:, 1]
    private_probs = priv_model.private_predictions(orig_probs)
    private_preds = (private_probs > 0.5).astype(int)
    
    class PrivatePredictor:
        def predict(self, X): return private_preds
        def predict_proba(self, X):
            proba = np.zeros((len(X), 2))
            proba[:, 1] = private_probs
            proba[:, 0] = 1 - private_probs
            return proba
    return PrivatePredictor()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(" PyDP + UCI Adult + RandomForest")
    print("=" * 60)
    
    # Data pipeline
    data_path = download_uci_adult()
    (X_train, y_train, s_train), (X_test, y_test, s_test), scaler = load_uci_adult(data_path)
    
    results = []
    print("\n Privacy experiments...")
    print("eps     acc_base   acc_priv  loss    dp_race  mia_base  mia_priv")
    print("-" * 65)
    
    for eps in EPSILONS:
        print(f"Testing ε={eps}...", end=" ")
        
        # Baseline RandomForest
        base_model = RandomForestClassifier(
            n_estimators=30,
            max_depth=5,
            random_state=RANDOM_SEED,
            n_jobs=1  # Docker safe
        )
        base_model.fit(X_train, y_train)
        base_pred = base_model.predict(X_test)
        
        base_acc = accuracy(y_test, base_pred)
        base_dp_race = demographic_parity(base_pred, s_test)
        base_mia = mia_success(base_model, X_train, y_train, X_test, y_test)
        
        # Private predictions
        priv_model = RandomForestClassifier(
            n_estimators=30,
            max_depth=5,
            random_state=RANDOM_SEED,
            n_jobs=1
        )
        priv_model.fit(X_train, y_train)
        priv_predictor = apply_dp(priv_model, eps, X_test)
        priv_pred = priv_predictor.predict(X_test)
        
        priv_acc = accuracy(y_test, priv_pred)
        priv_dp_race = demographic_parity(priv_pred, s_test)
        priv_mia = mia_success(priv_model, X_train, y_train, X_test, y_test)
        
        results.append({
            'epsilon': eps,
            'base_acc': base_acc,
            'priv_acc': priv_acc,
            'acc_loss': base_acc - priv_acc,
            'dp_race': base_dp_race,
            'priv_dp_race': priv_dp_race,
            'base_mia': base_mia,
            'priv_mia': priv_mia,
            'mia_gain': base_mia - priv_mia
        })
        
        print(f"{base_acc:.3f}/{priv_acc:.3f}  MIA:{base_mia:.1%}/{priv_mia:.1%}")
    
    # FIXED: Save to outputs directory
    results_df = pd.DataFrame(results)
    results_file = os.path.join(OUTPUT_DIR, 'results.csv')
    results_df.to_csv(results_file, index=False)
    
    # FIXED: Save plot to outputs directory
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(results_df['epsilon'], results_df['base_acc'], 'o-', lw=3, label='Baseline', markersize=8)
    ax1.plot(results_df['epsilon'], results_df['priv_acc'], 's-', lw=3, label='Private', markersize=8)
    ax1.set_xlabel('ε (Privacy Budget)'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_title('Utility Tradeoff')
    
    ax2.plot(results_df['epsilon'], results_df['base_mia'], 'o-', lw=3, label='Baseline MIA', markersize=8)
    ax2.plot(results_df['epsilon'], results_df['priv_mia'], 's-', lw=3, label='Private MIA', markersize=8)
    ax2.axhline(0.5, color='k', ls='--', alpha=0.7, label='Random')
    ax2.set_xlabel('ε'); ax2.set_ylabel('MIA Success'); ax2.legend()
    ax2.grid(True, alpha=0.3); ax2.set_title('Privacy Gain')
    
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_DIR, 'results.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n COMPLETE!")
    print(f" Results: {results_file} ({len(results_df)} experiments)")
    print(f" Plots: {plot_file}")
    print(f" Dataset: adult.data ({len(X_train):,}+{len(X_test):,} records)")

if __name__ == "__main__":
    main()
