import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(file_path='data/raw/asteroid_data.csv'):
    """
    Analyzes the dataset and saves basic plots to clarify patterns.
    """
    df = pd.read_csv(file_path)
    
    # 1. SUMMARY STATS
    print("\n--- DATA SUMMARY ---")
    print(df.describe())

    # 2. TARGET DISTRIBUTION
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Hazardous', data=df, palette='Set2')
    plt.title('Hazardous vs Safe Asteroids')
    plt.savefig('results/target_distribution.png')
    plt.close()

    # 3. CORRELATION HEATMAP (Sample of key features)
    keys = ['Absolute Magnitude', 'Est Dia in KM(min)', 'Relative Velocity km per sec', 'Miss Dist.(kilometers)']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[keys].corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('results/correlation_heatmap.png')
    plt.close()

    print("EDA Visuals saved to results/ folder.")
