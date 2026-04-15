import pandas as pd
import joblib
import os

def run_predictions(input_file='data/raw/asteroid_data.csv', model_file='results/asteroid_model.joblib'):
    """
    Loads a trained model, reads an input dataset,
    and saves the predictions to a CSV file.
    """
    # 1. LOAD THE MODEL
    if not os.path.exists(model_file):
        print(f"Error: Model not found at {model_file}. Please run 'python py/modeling.py' first.")
        return
    
    print(f"--- Loading Model: {model_file} ---")
    model = joblib.load(model_file)

    # 2. LOAD TEST DATASET
    # (Note: In a real scenario, this would be a separate 'test.csv')
    print(f"--- Loading Dataset: {input_file} ---")
    df = pd.read_csv(input_file)

    # Store original IDs for the final output (if they exist)
    ids = df['Neo Reference ID'] if 'Neo Reference ID' in df.columns else range(len(df))

    # --- REPLICATE CLEANING STEPS (Must match training exactly) ---
    cols_to_drop = [
        'Neo Reference ID', 'Name', 'Close Approach Date', 'Orbiting Body', 
        'Orbit Determination Date', 'Equinox', 'Hazardous', # Target dropped from X
        'Est Dia in KM(max)', 'Est Dia in M(min)', 'Est Dia in M(max)',
        'Est Dia in Miles(min)', 'Est Dia in Miles(max)',
        'Est Dia in Feet(min)', 'Est Dia in Feet(max)',
        'Relative Velocity km per hr', 'Miles per hour',
        'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)', 'Miss Dist.(miles)'
    ]
    
    # Filter only columns that exist in the dataframe
    X_new = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 3. USE MODEL TO PREDICT
    print("--- Running Predictions ---")
    predictions = model.predict(X_new)

    # 4. SAVE PREDICTIONS TO CSV
    results_df = pd.DataFrame({
        'Neo Reference ID': ids,
        'Predicted_Hazardous': predictions
    })

    output_path = 'results/predictions.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Predictions saved to: {output_path}")
    print(results_df.head())

if __name__ == "__main__":
    run_predictions()
