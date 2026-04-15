import os
from src.data_loader import load_and_preprocess
from src.eda import run_eda
from src.modeling import build_and_evaluate
from src.inverse_design import generate_ideal_scenarios
from src.visualizer import plot_feature_importance
from src.render_ui import generate_html_dashboard

def main():
    # 0. ENSURE RESULTS DIR EXISTS
    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. DATA PREP
    print("🚀 LOADING DATA...")
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # 2. EXPLORE DATA
    print("📊 ANALYZING DATA...")
    run_eda()

    # 3. TRAIN MODELS
    print("🤖 TRAINING MODELS...")
    best_model = build_and_evaluate(X_train, X_test, y_train, y_test)

    # 4. VISUALIZE
    print("📈 PLOTTING RESULTS...")
    plot_feature_importance(best_model, X_train.columns)

    # 5. INVERSE CHALLENGE
    print("🔍 SOLVING INVERSE CHALLENGE...")
    generate_ideal_scenarios(best_model, X_train)

    # 6. RENDER DASHBOARD
    print("🖥️ GENERATING PREVIEW DASHBOARD...")
    generate_html_dashboard()

    print("\n✅ PROJECT RUN COMPLETE!")
    print("Check the 'results/' folder for your submission files and dashboard.html!")

if __name__ == "__main__":
    main()
