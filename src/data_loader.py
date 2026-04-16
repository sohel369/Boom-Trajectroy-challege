import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess(train_path='data/raw/train.csv', labels_path='data/raw/train_labels.csv'):
    """
    Loads the Boom Challenge training data and labels, merges them,
    and splits into training and validation sets.
    """
    X_full = pd.read_csv(train_path)
    y_full = pd.read_csv(labels_path)
    
    # Check if they have the same number of rows
    if len(X_full) != len(y_full):
        print(f"Warning: Features ({len(X_full)}) and Labels ({len(y_full)}) row count mismatch!")

    # 1. HANDLE MISSING VALUES
    # (Checking for NaNs and filling with mean or median)
    X_full = X_full.fillna(X_full.mean())
    y_full = y_full.fillna(y_full.mean())

    # 2. SPLIT INTO TRAINING AND VALIDATION SETS
    # We use 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val

def load_test_data(test_path='data/raw/test.csv'):
    """
    Loads the test data for final submission.
    """
    X_test = pd.read_csv(test_path)
    X_test = X_test.fillna(X_test.mean())
    return X_test
