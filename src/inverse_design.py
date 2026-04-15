import pandas as pd
import numpy as np

def generate_ideal_scenarios(model, X_train, target_count=20):
    """
    Reverse-engineers asteroids that meet specific impact criteria.
    """
    print("\n--- INVERSE DESIGN MODULE ---")
    n = 20000 
    feature_names = X_train.columns.tolist()
    
    # 1. GENERATE RANDOM INPUTS FOR CORE PARAMETERS
    data = {
        'Absolute Magnitude': np.random.uniform(15, 25, n),
        'Est Dia in KM(min)': np.random.uniform(0.1, 1.5, n),
        'Relative Velocity km per sec': np.random.uniform(10, 45, n),
        'Miss Dist.(kilometers)': np.random.uniform(500000, 50000000, n),
        'Orbit Uncertainity': np.random.randint(0, 10, n),
        'Minimum Orbit Intersection': np.random.uniform(0, 0.4, n)
    }
    
    # Create simulator dataframe
    df_sim = pd.DataFrame(data)

    # 2. FILL OTHER FEATURES WITH MEANS FROM TRAINING DATA
    for col in feature_names:
        if col not in df_sim.columns:
            df_sim[col] = X_train[col].mean()

    # Reorder columns to match model training exactly
    df_sim = df_sim[feature_names]

    # 3. PREDICT OUTPUTS
    df_sim['Hazard_Risk'] = model.predict(df_sim)

    # 4. CALCULATE P80 AND R95 (Simulated Physics)
    df_sim['P80'] = df_sim['Relative Velocity km per sec'] * 1.5 + df_sim['Absolute Magnitude'] * 2
    df_sim['R95'] = df_sim['Est Dia in KM(min)'] * 100 + df_sim['Relative Velocity km per sec'] * 2

    # 5. FILTER BASED ON CHALLENGE CONDITIONS
    condition = (df_sim['P80'] >= 96) & (df_sim['P80'] <= 101) & (df_sim['R95'] <= 175)
    valid_scenarios = df_sim[condition]

    # 6. SAVE TOP 20
    final_output = valid_scenarios.head(target_count)
    final_output.to_csv('results/valid_scenarios.csv', index=False)
    
    print(f"Found {len(valid_scenarios)} matching scenarios.")
    print(f"Saved top 20 to results/valid_scenarios.csv")
