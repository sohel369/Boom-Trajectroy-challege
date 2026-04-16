import pandas as pd
import numpy as np
import json
import os

def generate_ideal_scenarios(model, X_train, target_count=20):
    """
    Simulates thousands of scenarios to find inputs that satisfy the buyer's constraints.
    """
    print("\n--- INVERSE DESIGN MODULE ---")
    
    # 1. LOAD CONSTRAINTS
    constraints_path = 'data/raw/constraints.json'
    if os.path.exists(constraints_path):
        with open(constraints_path, 'r') as f:
            config = json.load(f)
    else:
        # Fallback to defaults seen in constraints.json
        config = {
            "constraints": {"p80_min": 96.0, "p80_max": 101.0, "r95_max": 175.0},
            "input_bounds": {
                "energy": [0.5, 5.0], "angle_rad": [0.26, 1.57], "coupling": [0.2, 1.7],
                "strength": [0.4, 4.2], "porosity": [0.0, 0.33], "gravity": [1.02, 10.47],
                "atmosphere": [0.0, 1.0], "shape_factor": [0.7, 1.5]
            }
        }

    c = config['constraints']
    bounds = config['input_bounds']
    
    # 2. GENERATE RANDOM SAMPLES WITHIN BOUNDS
    n = 50000 
    df_sim = pd.DataFrame()
    
    for feat, b in bounds.items():
        # Handle both list and dict formats for bounds
        if isinstance(b, dict):
            df_sim[feat] = np.random.uniform(b['min'], b['max'], n)
        else:
            df_sim[feat] = np.random.uniform(b[0], b[1], n)
            
    # Ensure columns match training data order
    feature_names = X_train.columns.tolist()
    df_sim = df_sim[feature_names]

    # 3. PREDICT OUTPUTS USING TRAINED MODEL
    print(f"Simulating {n} scenarios and predicting outcomes...")
    preds = model.predict(df_sim)
    
    # Map predictions back to columns
    # Predictions structure: [P80, fines_frac, oversize_frac, R95, R50_fines, R50_oversize]
    target_names = ['P80', 'fines_frac', 'oversize_frac', 'R95', 'R50_fines', 'R50_oversize']
    for i, name in enumerate(target_names):
        df_sim[name] = preds[:, i]

    # 4. FILTER BASED ON CHALLENGE CONDITIONS
    condition = (df_sim['P80'] >= c['p80_min']) & \
                (df_sim['P80'] <= c['p80_max']) & \
                (df_sim['R95'] <= c['r95_max'])
    
    valid_scenarios = df_sim[condition]

    # 5. SAVE SUBMISSION
    if len(valid_scenarios) == 0:
        print("Warning: No scenarios met the constraints. Lowering selectivity for demonstration.")
        # Just take the ones closest to the range
        valid_scenarios = df_sim.copy()
        valid_scenarios['score'] = np.abs(valid_scenarios['P80'] - (c['p80_min'] + c['p80_max'])/2)
        valid_scenarios = valid_scenarios.sort_values('score')
    
    final_output = valid_scenarios.head(target_count)
    
    # Save the full results and also a submission-style CSV
    final_output.to_csv('results/valid_scenarios.csv', index=False)
    
    # Submission template usually only needs inputs
    submission = final_output[feature_names]
    submission.to_csv('results/design_submission.csv', index=False)
    
    print(f"Found {len(valid_scenarios)} matching scenarios.")
    print(f"Saved submission to results/design_submission.csv")
