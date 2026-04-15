import pandas as pd
import os

def load_and_prepare_data(file_path='data/raw/asteroid_data.csv'):
    """
    Loads the asteroid dataset, prints exploration info, 
    and splits into features (X) and target (y).
    """
    print(f"Loading data from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run 'python src/download_data.py' first.")
        return None, None

    # 1. Load the dataset using pandas
    df = pd.read_csv(file_path)

    # 2. Print first 5 rows
    print("\n--- First 5 Rows ---")
    print(df.head())

    # 3. Show column names
    print("\n--- Column Names ---")
    print(df.columns.tolist())

    # 4. Column Explanations (Summary)
    # Absolute Magnitude: Brightness (proxy for size)
    # Est Dia in KM: Size estimate
    # Relative Velocity: Speed relative to Earth
    # Miss Dist: Distance from Earth at closest approach
    # Orbit Uncertainity: Precision of path prediction (0-9)
    # Hazardous: Target classification

    # 5. Separate input features (X) and output targets (y)
    # Selecting key physical and orbital features
    features = [
        'Absolute Magnitude', 
        'Est Dia in KM(min)', 
        'Est Dia in KM(max)', 
        'Relative Velocity km per sec', 
        'Miss Dist.(kilometers)', 
        'Orbit Uncertainity', 
        'Minimum Orbit Intersection'
    ]

    X = df[features]
    y = df['Hazardous']

    print("\n--- Data Preparation Complete ---")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    
    return X, y

if __name__ == "__main__":
    X, y = load_and_prepare_data()
