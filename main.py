import os
from src.data_loader import load_and_preprocess, load_test_data
from src.eda import run_eda
from src.modeling import build_and_evaluate
from src.inverse_design import generate_ideal_scenarios
from src.visualizer import plot_feature_importance
from src.render_ui import generate_html_dashboard

def main():
    # 0. ENSURE DIRS EXIST
    for d in ['results', 'data/processed']:
        if not os.path.exists(d):
            os.makedirs(d)

    # 1. DATA PREP
    print("🚀 LOADING BUYER DATASETS (TRAIN & LABELS)...")
    X_train, X_val, y_train, y_val = load_and_preprocess()

    # 2. EXPLORE DATA
    print("📊 ANALYZING DATA DISTRIBUTIONS...")
    run_eda()

    # 3. TRAIN MODELS (Multi-Output Regression)
    print("🤖 TRAINING PREDICTION MODEL...")
    model = build_and_evaluate(X_train, X_val, y_train, y_val)

    # 4. VISUALIZE
    print("📈 PLOTTING FEATURE IMPORTANCE...")
    plot_feature_importance(model, X_train.columns)

    # 5. TEST PREDICTIONS (Optional: for leaderboard)
    print("📋 GENERATING TEST PREDICTIONS...")
    X_test = load_test_data()
    test_preds = model.predict(X_test)
    # Save test predictions in the format requested by template
    # Usually: inputs + preds
    target_names = ['P80', 'fines_frac', 'oversize_frac', 'R95', 'R50_fines', 'R50_oversize']
    test_results = X_test.copy()
    for i, name in enumerate(target_names):
        test_results[name] = test_preds[:, i]
    test_results.to_csv('results/test_predictions.csv', index=False)

    # 6. INVERSE CHALLENGE (Solve for constraints)
    print("🔍 SOLVING INVERSE DESIGN CHALLENGE...")
    generate_ideal_scenarios(model, X_train)

    # 7. RENDER DASHBOARD
    print("🖥️ GENERATING PREVIEW DASHBOARD...")
    generate_html_dashboard()

    print("\n✅ PROJECT RUN COMPLETE!")
    print("Check the 'results/' folder for your submission files:")
    print("1. results/test_predictions.csv  <- Forward predictions")
    print("2. results/design_submission.csv <- Inverse design solution")
    print("3. results/dashboard.html        <- Visual verification")

if __name__ == "__main__":
    main()
