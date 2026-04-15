# ☄️ Asteroid Impact Debris Prediction

A machine learning project designed to classify asteroid hazards and simulate impact debris scenarios using NASA Near-Earth Object (NEO) data.

## 📌 Project Overview
This project uses physical and orbital characteristics of asteroids (like velocity, diameter, and miss distance) to predict whether an object is potentially hazardous to Earth. Additionally, it features a simulation tool to identify specific "ideal" asteroid scenarios that meet target impact intensity and debris radius criteria.

## 🚀 Approach & Modeling
- **Data Source**: NASA Open Data (Near-Earth Objects).
- **Preprocessing**: Cleans redundant units (meters, miles) and removes non-physical identifiers (Name, ID).
- **Classification Model**: **Random Forest Classifier**.
  - Chosen for its high accuracy (~99%) and ability to handle the complex, non-linear relationships found in orbital mechanics.
- **Scenario Simulation**: Uses a Monte Carlo approach to generate thousands of random asteroid inputs and filter for those satisfying specific physical constraints (P80 impact intensity and R95 debris radius).

## 🛠️ How to Run
1. **Setup Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Download Data**:
   ```powershell
   python src/download_data.py
   ```
3. **Analyze Data**:
   ```powershell
   python src/data_analysis.py
   ```
4. **Train Model**:
   ```powershell
   python py/modeling.py
   ```
5. **Generate Scenarios**:
   ```powershell
   python py/scenario_generator.py
   ```

## ⚠️ Limitations
- **Imbalanced Data**: The dataset contains far more safe asteroids than hazardous ones, which can bias simple models.
- **Simulated Metrics**: The P80 and R95 metrics in the scenario generator are currently based on simplified physics formulas for demonstration purposes.
- **Atmospheric Variables**: This model does not currently account for Earth's atmospheric composition or asteroid density (composition).

## 🔮 Future Improvements
- **Feature Engineering**: Add "Kinetic Energy" as a derived feature (Mass x Velocity²).
- **Advanced Models**: Implement XGBoost or Neural Networks to see if they outperform the Random Forest.
- **Handling Imbalance**: Use SMOTE (Synthetic Minority Over-sampling Technique) to balance the classification target.
- **Real-time API**: Connect to NASA's NeoWS API for live asteroid tracking.
