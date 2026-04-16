import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(train_path='data/raw/train.csv', labels_path='data/raw/train_labels.csv'):
    """
    Analyzes the dataset and saves plots that help understand feature relationships.
    """
    X = pd.read_csv(train_path)
    y = pd.read_csv(labels_path)
    
    # Merge for correlation analysis
    df = pd.concat([X, y], axis=1)
    
    # 1. SUMMARY STATS
    print("\n--- DATA SUMMARY ---")
    print(df.describe())

    # 2. FEATURE DISTRIBUTIONS (Top 4 inputs)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    top_features = ['energy', 'porosity', 'strength', 'gravity']
    for i, col in enumerate(top_features):
        ax = axes[i//2, i%2]
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('results/feature_distributions.png')
    plt.close()

    # 3. CORRELATION HEATMAP
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature and Target Correlation Heatmap')
    plt.savefig('results/correlation_heatmap.png')
    plt.close()

    print("EDA Visuals saved to results/ folder.")
