# run_simple.py
import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Running simplified healthcare fraud detection...")

try:
    # Test imports
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTENC
    import yaml
    
    print("✓ All required packages imported successfully!")
    
    # Check for config file
    if os.path.exists("config.yaml"):
        print("✓ Config file found")
    else:
        print("✗ Config file not found - creating default...")
        # Create default config
        default_config = """
paths:
  train_data: "data/raw/Train.csv"
  beneficiary_data: "data/raw/Beneficiary.csv"
  inpatient_data: "data/raw/Inpatient.csv"
  outpatient_data: "data/raw/Outpatient.csv"
  output_dir: "data/processed/"
  model_dir: "models/"

preprocessing:
  numerical_imputation: "median"
  categorical_imputation: "mode"
  test_size: 0.2
  random_state: 42

smote:
  sampling_strategy: "auto"
  k_neighbors: 5
  random_state: 42

models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    random_state: 42
  logistic_regression:
    C: 1.0
    max_iter: 1000
    random_state: 42

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
  cv_folds: 5
"""
        with open("config.yaml", "w") as f:
            f.write(default_config)
        print("✓ Default config created")
    
    # Check for data directory
    if not os.path.exists("data/raw"):
        print("Creating data directories...")
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        print("✓ Data directories created")
    
    print("\n" + "="*60)
    print("SETUP VERIFICATION COMPLETE")
    print("="*60)
    print("\nTo run the project:")
    print("1. Download Kaggle dataset and place in data/raw/")
    print("2. Run: python main.py --mode preprocess")
    print("3. Run: python main.py --mode train")
    print("4. Run: python main.py --mode dashboard")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install required packages:")
    print("pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn plotly streamlit pyyaml")