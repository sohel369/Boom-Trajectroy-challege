# Project Setup Guide: Asteroid Impact Debris Prediction

Welcome to your Machine Learning project! This guide will help you set up your development environment on Windows.

## 1. Install Python
If you don't have Python installed yet, follow these steps:
1. Go to [python.org](https://www.python.org/downloads/windows/).
2. Download the latest stable version (e.g., Python 3.12 or 3.13).
3. **Important:** When running the installer, make sure to check the box that says **"Add Python to PATH"**.
4. Click "Install Now".

> [!NOTE]
> I detected that Python 3.13.6 is already installed on your system! You can skip this step unless you want a different version.

## 2. Create a Virtual Environment
A virtual environment keeps your project dependencies isolated.
1. Open your terminal (PowerShell or Command Prompt) in the project folder.
2. Run the following command:
   ```powershell
   python -m venv venv
   ```
   This creates a folder named `venv` in your project directory.

## 3. Activate the Environment & Install Libraries
You must activate the environment every time you start working on the project.

### Activation:
- **PowerShell:**
  ```powershell
  .\venv\Scripts\activate
  ```
- **Command Prompt:**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

### Installation:
Once activated, you will see `(venv)` in your terminal prompt. Now install the required libraries:
```powershell
pip install -r requirements.txt
```
This will install `numpy`, `pandas`, `scikit-learn`, and `matplotlib` (along with `seaborn` and `jupyter` which I've added as they are very useful for ML).

## 4. Project Folder Structure
I have already created a clean structure for you:

```text
AI challenge/
├── data/               # For your datasets
│   ├── raw/            # Untouched original data
│   └── processed/      # Cleaned data ready for ML
├── notebooks/          # For Jupyter notebooks (.ipynb)
├── src/                # For reusable Python scripts (.py)
├── results/            # For plots, logs, and saved models
├── requirements.txt    # List of project dependencies
└── README.md           # Project overview
```

## Next Steps
- **Get Data:** Look for asteroid datasets (e.g., from NASA's CNEOS or OpenData).
- **EDA:** Create a notebook in `notebooks/` to explore your data.
- **Model:** Start building your prediction algorithms in `src/`.
