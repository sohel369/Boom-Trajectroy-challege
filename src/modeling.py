import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def build_and_evaluate(X_train, X_val, y_train, y_val):
    """
    Trains a Multi-Output Random Forest Regressor and evaluates performance.
    """
    print("\n--- MODEL TRAINING (REGRESSION) ---")

    # The buyer's data has multiple targets: P80, R95, etc.
    # RandomForestRegressor can handle multiple outputs directly.
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("Fitting model on training data...")
    model.fit(X_train, y_train)

    # 1. EVALUATE
    print("\nTraining complete. Evaluating on validation set...")
    y_pred = model.predict(X_val)
    
    # Calculate metrics for each target
    targets = y_train.columns
    for i, target in enumerate(targets):
        mae = mean_absolute_error(y_val.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_val.iloc[:, i], y_pred[:, i])
        print(f"Target [{target}]: MAE = {mae:.4f}, R2 Score = {r2:.4f}")

    # 2. SAVE BEST
    joblib.dump(model, 'results/best_model.joblib')
    print("\nBest model saved to results/best_model.joblib")
    
    return model
