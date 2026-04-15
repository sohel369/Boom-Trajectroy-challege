import pandas as pd
import numpy as np
import joblib
import os

def generate_scenarios(n_samples=5000, target_count=20):
    """
    1. Generates random asteroid features.
    2. Uses trained model to predict risk.
    3. Calculates physics metrics (P80, R95) via formulas.
    4. Filters scenarios that meet specific conditions.
    """
    
    # 1. LOAD MODEL
    model_path = 'results/asteroid_model.joblib'
    if not os.path.exists(model_path):
        print("Error: Model not found. Run modeling.py first!")
        return

    model = joblib.load(model_path)

    # 2. GENERATE RANDOM INPUTS (Based on data bounds)
    # We define reasonable ranges for our features
    print(f"Generating {n_samples} random scenarios...")
    data = {
        'Absolute Magnitude': np.random.uniform(15, 30, n_samples),
        'Est Dia in KM(min)': np.random.uniform(0.01, 2.0, n_samples),
        'Relative Velocity km per sec': np.random.uniform(1, 40, n_samples),
        'Miss Dist.(kilometers)': np.random.uniform(1000000, 70000000, n_samples),
        'Orbit Uncertainity': np.random.randint(0, 10, n_samples),
        'Minimum Orbit Intersection': np.random.uniform(0, 0.5, n_samples)
    }
    df_scenarios = pd.DataFrame(data)

    # 3. PREDICT HAZARD
    df_scenarios['Hazard_Prediction'] = model.predict(df_scenarios)

    # 4. CALCULATE SCENARIO METRICS (P80 and R95)
    # Note: These are simulated formulas for learning purposes
    # P80: Impact Intensity (Simulated)
    df_scenarios['P80'] = (df_scenarios['Relative Velocity km per sec'] * 1.5 + 
                           df_scenarios['Absolute Magnitude'] * 2)
    
    # R95: Debris Radius (Simulated)
    df_scenarios['R95'] = (df_scenarios['Est Dia in KM(min)'] * 100 + 
                           df_scenarios['Relative Velocity km per sec'] * 2)

    # 5. FILTER BASED ON CONDITIONS
    # Conditions: P80 [96, 101], R95 <= 175
    mask = (df_scenarios['P80'] >= 96) & \
           (df_scenarios['P80'] <= 101) & \
           (df_scenarios['R95'] <= 175)
    
    valid_scenarios = df_scenarios[mask]

    # 6. SAVE RESULTS
    count = len(valid_scenarios)
    print(f"Found {count} scenarios matching conditions.")

    if count >= target_count:
        final_scenarios = valid_scenarios.head(target_count)
        output_file = 'results/valid_scenarios.csv'
        final_scenarios.to_csv(output_file, index=False)
        print(f"✅ Successfully saved 20 scenarios to {output_file}")
        print("\nPreview of Valid Scenarios:")
        print(final_scenarios[['P80', 'R95', 'Hazard_Prediction']].head())
    else:
        print(f"Only found {count} results. Try increasing n_samples.")

if __name__ == "__main__":
    generate_scenarios()
