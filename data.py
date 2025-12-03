"""UCI Adult dataset loader"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import Config

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

def load_uci_adult(data_path="adult.data"):
    """Load and preprocess UCI Adult dataset."""
    print(f"Loading UCI Adult from: {data_path}")
    
    # Load raw data
    df = pd.read_csv(data_path, names=COLUMN_NAMES, na_values=" ?", skipinitialspace=True)
    
    # Drop missing values
    df = df.dropna()
    print(f"Dataset shape: {df.shape}")
    
    # Target: income binary (1 = >50K)
    df["income"] = (df["income"] == ">50K").astype(int)
    
    # Features (numeric + categorical)
    numeric_features = ["age", "fnlwgt", "education_num", "capital_gain", 
                       "capital_loss", "hours_per_week"]
    categorical_features = ["workclass", "education", "marital_status", 
                           "occupation", "relationship", "race", "sex", "native_country"]
    
    # Encode categoricals
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Prepare features
    features = numeric_features + categorical_features
    X = df[features].values
    y = df["income"].values
    sensitive = df[["race", "sex"]].values
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=Config.TEST_SPLIT, random_state=Config.RANDOM_SEED, stratify=y
    )
    
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Positive rate: {y.mean():.1%}")
    
    return (X_train, y_train, s_train), (X_test, y_test, s_test), scaler

if __name__ == "__main__":
    from download_data import download_uci_adult
    data_path = download_uci_adult()
    load_uci_adult(data_path)
