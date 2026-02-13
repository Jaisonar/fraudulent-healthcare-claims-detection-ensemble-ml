# text_architecture.py
def create_text_architecture():
    architecture = """
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │               HEALTHCARE FRAUD DETECTION SYSTEM ARCHITECTURE               │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                             USERS & STAKEHOLDERS                            │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  • Insurance Analysts and Investigators                                     │
    │  • System Administrators                                                    │
    └──────────────────────┬──────────────────────────┬──────────────────────────┘
                           │                          │
                           ▼                          ▼
    ┌──────────────────────┴──────────────────────────┴──────────────────────────┐
    │                            PRESENTATION LAYER                              │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  ┌──────────────────────┐                    ┌──────────────────────┐      │
    │  │   Streamlit Dashboard │                    │      REST API        │      │
    │  │  • User Interface    │                    │  • System Integration │      │
    │  │  • Real-time Views   │                    │  • External Services  │      │
    │  └──────────┬───────────┘                    └──────────┬───────────┘      │
    └─────────────┼───────────────────────────────────────────┼──────────────────┘
                  │                                           │
                  ▼                                           ▼
    ┌─────────────┴───────────────────────────────────────────┴──────────────────┐
    │                            APPLICATION LAYER                               │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
    │  │  Orchestrator │  │  Preprocessor │  │    Trainer    │  │  Evaluator   │    │
    │  │   (main.py)   │  │     Engine    │  │    Engine     │  │    Engine    │    │
    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
    └─────────┼──────────────────┼─────────────────┼─────────────────┼────────────┘
              │                  │                 │                 │
              ▼                  ▼                 ▼                 ▼
    ┌─────────┴──────────────────┴─────────────────┴─────────────────┴────────────┐
    │                                DATA LAYER                                   │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  ┌─────────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
    │  │   Raw Data Storage   │  │  Processed Data │  │  Model Storage   │         │
    │  │  • Kaggle CSV files  │  │  • Train/Test   │  │  • Pickle files  │         │
    │  │  • Beneficiary data  │  │  • Features     │  │  • XGBoost       │         │
    │  │  • Inpatient/Outpatient│  │  • Encodings   │  │  • Random Forest │         │
    │  └──────────┬──────────┘  └────────┬────────┘  └────────┬────────┘         │
    └─────────────┼───────────────────────┼────────────────────┼──────────────────┘
                  │                       │                    │
                  └───────────────────────┼────────────────────┘
                                          │
                                          ▼
    ┌─────────────────────────────────────┼──────────────────────────────────────┐
    │                       MACHINE LEARNING LAYER                              │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
    │  │   SMOTENC     │  │ Random Forest │  │   XGBoost    │  │    Logistic   │    │
    │  │  • Class      │  │  • Ensemble   │  │  • Gradient   │  │  • Baseline   │    │
    │  │  Balancing    │  │  Method       │  │  Boosting    │  │  Model       │    │
    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
    │         └──────────────────┴─────────────────┴─────────────────┘            │
    │                                    │                                        │
    │                                    ▼                                        │
    │                          ┌──────────────────┐                               │
    │                          │    Ensemble      │                               │
    │                          │  • Voting/Stacking│                               │
    │                          │  • Weighted Avg  │                               │
    │                          └────────┬─────────┘                               │
    └───────────────────────────────────┼─────────────────────────────────────────┘
                                        │
                                        ▼
    ┌───────────────────────────────────┴─────────────────────────────────────────┐
    │                             OUTPUT & MONITORING                             │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  • Fraud Predictions (0/1 labels)                                          │
    │  • Probability Scores                                                      │
    │  • Performance Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)          │
    │  • Visualizations (Confusion Matrix, ROC Curve, Feature Importance)        │
    │  • Reports & Analytics                                                     │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    DATA FLOW: User → Dashboard → Orchestrator → Preprocessor → Trainer → Evaluator → Output
    ML FLOW: Raw Data → SMOTENC → Models → Ensemble → Predictions
    """
    
    print(architecture)
    
    # Save to file
    with open('architecture_diagram.txt', 'w') as f:
        f.write(architecture)
    
    print("✓ Text architecture saved to 'architecture_diagram.txt'")
    
    # Also create a simplified version for documentation
    simplified = """
    ==============================================================================
                         SYSTEM ARCHITECTURE OVERVIEW
    ==============================================================================
    
    [USERS]
        │
        ▼
    [PRESENTATION LAYER]
    ├── Streamlit Dashboard (User Interface)
    └── REST API (Optional Integration)
        │
        ▼
    [APPLICATION LAYER]
    ├── main.py (Orchestrator)
    ├── data_preprocessing.py
    ├── model_training.py
    └── model_evaluation.py
        │
        ▼
    [DATA LAYER]
    ├── Raw CSV Files (Kaggle Dataset)
    ├── Processed Train/Test Data
    ├── Trained Models (Pickle)
    └── Results & Metrics (JSON/CSV)
        │
        ▼
    [MACHINE LEARNING LAYER]
    ├── SMOTENC (Class Balancing)
    ├── Random Forest Classifier
    ├── XGBoost Classifier
    ├── Logistic Regression
    └── Voting/Stacking Ensemble
        │
        ▼
    [OUTPUT]
    ├── Fraud Predictions
    ├── Performance Metrics
    ├── Visualizations
    └── Reports
    
    ==============================================================================
                            KEY DATA FLOWS
    ==============================================================================
    1. Data Pipeline: Raw CSV → Preprocessing → SMOTENC → Feature Engineering
    2. Training Pipeline: Processed Data → Model Training → Model Evaluation
    3. Prediction Pipeline: New Data → Preprocessing → Model Inference → Results
    4. Monitoring: All Processes → Logging → Metrics → Dashboard
    
    ==============================================================================
                            TECHNOLOGY STACK
    ==============================================================================
    • Programming: Python 3.8+
    • ML Libraries: scikit-learn, XGBoost, imbalanced-learn (SMOTENC)
    • Data Processing: pandas, numpy
    • Visualization: Plotly, matplotlib
    • Dashboard: Streamlit
    • Model Storage: pickle
    • Configuration: YAML
    """
    
    with open('architecture_overview.txt', 'w') as f:
        f.write(simplified)
    
    print("✓ Simplified architecture saved to 'architecture_overview.txt'")

if __name__ == "__main__":
    create_text_architecture()