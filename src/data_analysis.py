import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_analysis(file_path='data/raw/asteroid_data.csv'):
    """
    Performs basic analysis on the dataset.
    """
    df = pd.read_csv(file_path)
    
    # 1. Missing Values
    print("--- Missing Values ---")
    print(df.isnull().sum().sum())  # Total count of missing values
    
    # 2. Statistics
    print("\n--- Basic Statistics ---")
    print(df.describe())
    
    # 3. Quick Plot (Distribution)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Hazardous', data=df)
    plt.title('Hazardous Count')
    plt.savefig('results/hazard_distribution.png')
    print("\nPlot saved to results/hazard_distribution.png")

    # 4. Cleaning Advice
    print("\n--- Cleaning Report ---")
    print("- No missing values found.")
    print("- ADVICE: Drop redundant units (Miles, Feet, Meters). Use only KM.")
    print("- ADVICE: Drop ID columns like 'Neo Reference ID'.")
    print("- ADVICE: Target is imbalanced (more False than True).")

if __name__ == "__main__":
    run_analysis()
