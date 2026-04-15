import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def build_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains two models and chooses the best one for output.
    """
    print("\n--- MODEL TRAINING ---")

    # 1. LOGISTIC REGRESSION (Linear)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    print(f"Linear (Logistic) Accuracy: {lr_acc:.2%}")

    # 2. RANDOM FOREST (Non-Linear)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest Accuracy:     {rf_acc:.2%}")

    # 3. COMPARE AND SAVE BEST
    if rf_acc >= lr_acc:
        print("\nWinner: RANDOM FOREST (Recommended for submission)")
        best_model = rf
    else:
        print("\nWinner: LOGISTIC REGRESSION")
        best_model = lr

    # Save to disk
    joblib.dump(best_model, 'results/best_model.joblib')
    print("Best model saved to results/best_model.joblib")
    
    return best_model
