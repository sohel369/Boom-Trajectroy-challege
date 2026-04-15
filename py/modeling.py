import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def build_models(file_path='data/raw/asteroid_data.csv'):
    """
    Loads data, cleans it, trains a Logistic Regression and Random Forest,
    and compares their performance.
    """
    print(f"--- Loading data from {file_path} ---")
    df = pd.read_csv(file_path)

    # --- DATA CLEANING (Essential for ML) ---
    # Drop IDs and redundant units (Meters, Miles, Feet)
    cols_to_drop = [
        'Neo Reference ID', 'Name', 'Close Approach Date', 'Orbiting Body', 
        'Orbit Determination Date', 'Equinox', 
        'Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)',
        'Est Dia in Miles(min)', 'Est Dia in Miles(max)',
        'Est Dia in Feet(min)', 'Est Dia in Feet(max)',
        'Relative Velocity km per hr', 'Miles per hour',
        'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)', 'Miss Dist.(miles)'
    ]
    df_clean = df.drop(columns=cols_to_drop)

    # --- 1. SEPARATE FEATURES (X) AND TARGET (y) ---
    X = df_clean.drop('Hazardous', axis=1)
    y = df_clean['Hazardous']

    # --- 2. TRAIN-TEST SPLIT ---
    # 80% for training, 20% for final testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # --- 3. TRAIN MODELS ---
    
    # Model 1: Logistic Regression (Linear Model for classification)
    # Note: We use Logistic instead of Linear because our target is True/False
    print("\n--- Training Logistic Regression (Linear Approach) ---")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    # Model 2: Random Forest (Non-Linear Approach)
    print("--- Training Random Forest Classifier ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # --- 4. COMPARE PERFORMANCE ---
    lr_acc = accuracy_score(y_test, lr_preds)
    rf_acc = accuracy_score(y_test, rf_preds)

    print("\n--- RESULTS COMPARISON ---")
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(f"Random Forest Accuracy:       {rf_acc:.4f}")

    # --- 5. SAVE THE BEST MODEL ---
    import joblib
    model_filename = 'results/asteroid_model.joblib'
    joblib.dump(rf_model, model_filename)
    print(f"\n✅ Best model (Random Forest) saved to {model_filename}")

if __name__ == "__main__":
    build_models()
