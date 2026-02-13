# main.py
import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Healthcare Fraud Detection System')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'evaluate', 'dashboard'], 
                       default='dashboard', help='Run mode')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.mode == 'preprocess':
        print("Running data preprocessing...")
        try:
            from src.data_preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor(args.config)
            X_train, X_test, y_train, y_test, scaler, label_encoders = preprocessor.run_pipeline()
            print("✓ Data preprocessing completed successfully!")
        except Exception as e:
            print(f"✗ Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
        
    elif args.mode == 'train':
        print("Training models...")
        try:
            from src.model_training import ModelTrainer
            trainer = ModelTrainer(args.config)
            predictions = trainer.train_all_models(trainer.config['paths']['output_dir'])
            trainer.save_models()
            trainer.save_all_results()
            print("✓ Model training completed successfully!")
        except Exception as e:
            print(f"✗ Error in training: {e}")
            import traceback
            traceback.print_exc()
        
    elif args.mode == 'evaluate':
        print("Evaluating models...")
        try:
            from src.model_evaluation import ModelEvaluator
            from src.model_training import ModelTrainer
            from src.project_utils import load_model  # Changed
            
            trainer = ModelTrainer(args.config)
            evaluator = ModelEvaluator(args.config)
            
            # Load data
            data_path = trainer.config['paths']['output_dir']
            X_train, X_test, y_train, y_test = trainer.load_data(data_path)
            
            # Load models
            models = {
                'Random Forest': load_model('random_forest'),
                'XGBoost': load_model('xgboost'),
                'Logistic Regression': load_model('logistic_regression')
            }
            
            # Run evaluation
            evaluation_results = evaluator.run_complete_evaluation(models, X_test, y_test, X_train, y_train)
            print("✓ Model evaluation completed successfully!")
        except Exception as e:
            print(f"✗ Error in evaluation: {e}")
            import traceback
            traceback.print_exc()
        
    elif args.mode == 'dashboard':
        print("Starting dashboard...")
        try:
            import subprocess
            subprocess.run(["streamlit", "run", "src/streamlit_app.py"])
        except Exception as e:
            print(f"✗ Error starting dashboard: {e}")
            print("\nYou can also run: streamlit run src/streamlit_app.py")
        
    else:
        print("Invalid mode selected")

if __name__ == "__main__":
    main()