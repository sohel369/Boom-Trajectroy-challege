import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_importance(model, features):
    """
    Shows which features the Random Forest found most important.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance (Which factor matters most?)')
        plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.savefig('results/feature_importance.png')
        plt.close()
        print("Feature importance graph saved to results/.")
    else:
        print("Linear model used - skipping feature importance plot.")
