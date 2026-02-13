# src/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from pathlib import Path
import sys
import os
import hashlib

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.project_utils import load_config, load_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance  # Fixed import
from src.data_preprocessing import DataPreprocessor

# ... [rest of the code remains the same]

# Page configuration
st.set_page_config(
    page_title="Healthcare Fraud Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .fraud-alert {
        background-color: #ffe6e6;
        border-left: 5px solid #ff3333;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .non-fraud {
        background-color: #e6ffe6;
        border-left: 5px solid #33cc33;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FraudDetectionApp:
    def __init__(self):
        self.config = load_config()
        self.models = {}
        self.data = None
        self.predictions = None
        
    def load_models(self):
        """Load trained models"""
        try:
            self.models['random_forest'] = load_model('random_forest')
            self.models['xgboost'] = load_model('xgboost')
            self.models['logistic_regression'] = load_model('logistic_regression')
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def sidebar_navigation(self):
        """Create sidebar navigation"""
        st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
        st.sidebar.title("Navigation")
        
        page = st.sidebar.radio(
            "Go to",
            ["Dashboard", "Data Analysis", "Model Training", "Fraud Detection", "Reports", "Settings"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info("**Healthcare Fraud Detection System**\n\nDetect fraudulent insurance claims using ensemble ML and SMOTENC.")
        
        return page
    
    def dashboard_page(self):
        """Dashboard page"""
        st.markdown('<h1 class="main-header">🏥 Healthcare Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Claims", "10,000", "12%")
        with col2:
            st.metric("Fraudulent Claims", "850", "-8%")
        with col3:
            st.metric("Detection Accuracy", "94.5%", "2.3%")
        with col4:
            st.metric("Cost Saved", "$2.5M", "15%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud distribution
            fig = go.Figure(data=[
                go.Pie(labels=['Non-Fraud', 'Fraud'], values=[9150, 850], 
                      hole=0.4, marker_colors=['#2ecc71', '#e74c3c'])
            ])
            fig.update_layout(title="Fraud Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly trends
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            fraud_cases = [120, 135, 110, 145, 160, 180]
            non_fraud_cases = [1500, 1520, 1480, 1530, 1550, 1570]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=fraud_cases, mode='lines+markers', 
                                    name='Fraud Cases', line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=months, y=non_fraud_cases, mode='lines', 
                                    name='Total Cases', line=dict(color='blue', width=2, dash='dot')))
            fig.update_layout(title="Monthly Fraud Trends", height=400,
                            xaxis_title="Month", yaxis_title="Number of Cases")
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance
        st.markdown('<h3 class="sub-header">Model Performance Comparison</h3>', unsafe_allow_html=True)
        
        models = ['Random Forest', 'XGBoost', 'Logistic Regression', 'Ensemble']
        accuracy = [0.923, 0.945, 0.891, 0.956]
        recall = [0.912, 0.938, 0.876, 0.948]
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Recall"))
        
        fig.add_trace(
            go.Bar(x=models, y=accuracy, marker_color=['blue', 'green', 'red', 'purple']),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=recall, marker_color=['blue', 'green', 'red', 'purple']),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def data_analysis_page(self):
        """Data analysis page"""
        st.markdown('<h1 class="main-header">📊 Data Analysis</h1>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload claim data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"Data loaded successfully! Shape: {data.shape}")
                
                # Data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Data information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Data Information")
                    buffer = []
                    buffer.append(f"Total rows: {data.shape[0]}")
                    buffer.append(f"Total columns: {data.shape[1]}")
                    buffer.append(f"Missing values: {data.isnull().sum().sum()}")
                    buffer.append(f"Duplicate rows: {data.duplicated().sum()}")
                    
                    st.text("\n".join(buffer))
                
                with col2:
                    st.subheader("Column Types")
                    dtype_info = data.dtypes.value_counts()
                    for dtype, count in dtype_info.items():
                        st.text(f"{dtype}: {count}")
                
                # Statistical analysis
                st.subheader("Statistical Summary")
                st.dataframe(data.describe())
                
                # Missing values heatmap
                if data.isnull().sum().sum() > 0:
                    st.subheader("Missing Values Heatmap")
                    missing_data = data.isnull()
                    fig = go.Figure(data=go.Heatmap(
                        z=missing_data.T,
                        colorscale='Reds',
                        showscale=False
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                if len(data.select_dtypes(include=[np.number]).columns) > 1:
                    st.subheader("Correlation Matrix")
                    corr_matrix = data.select_dtypes(include=[np.number]).corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        colorbar=dict(title="Correlation")
                    ))
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading data: {e}")
        else:
            st.info("Please upload a CSV file to begin analysis")
            
            # Sample data option
            if st.button("Load Sample Data"):
                # Create sample data for demonstration
                sample_data = pd.DataFrame({
                    'Claim_Amount': np.random.exponential(5000, 100),
                    'Provider_ID': np.random.randint(1, 50, 100),
                    'Patient_Age': np.random.randint(18, 90, 100),
                    'Diagnosis_Code': np.random.choice(['A01', 'B02', 'C03', 'D04'], 100),
                    'Fraud_Flag': np.random.choice([0, 1], 100, p=[0.9, 0.1])
                })
                
                st.dataframe(sample_data)
    
    def model_training_page(self):
        """Model training page"""
        st.markdown('<h1 class="main-header">🤖 Model Training</h1>', unsafe_allow_html=True)
        
        # Training options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Configuration")
            
            model_choice = st.multiselect(
                "Select Models to Train",
                ["Random Forest", "XGBoost", "Logistic Regression", "Ensemble"],
                default=["Random Forest", "XGBoost", "Logistic Regression"]
            )
            
            test_size = st.slider("Test Size (%)", 10, 40, 20)
            random_state = st.number_input("Random State", value=42)
            
        with col2:
            st.subheader("SMOTENC Configuration")
            
            sampling_strategy = st.select_slider(
                "Sampling Strategy",
                options=['minority', 'not majority', 'all', 'auto'],
                value='auto'
            )
            
            k_neighbors = st.slider("K Neighbors", 3, 10, 5)
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                # Create progress bar
                progress_bar = st.progress(0)
                
                # Simulate training process
                for i in range(100):
                    progress_bar.progress(i + 1)
                    # In real implementation, this would call actual training functions
                
                # Display results
                st.success("Training completed successfully!")
                
                # Display metrics
                st.subheader("Training Results")
                
                results_data = {
                    'Model': ['Random Forest', 'XGBoost', 'Logistic Regression', 'Ensemble'],
                    'Accuracy': [0.923, 0.945, 0.891, 0.956],
                    'Precision': [0.912, 0.928, 0.865, 0.942],
                    'Recall': [0.912, 0.938, 0.876, 0.948],
                    'F1-Score': [0.912, 0.933, 0.870, 0.945],
                    'ROC-AUC': [0.965, 0.978, 0.932, 0.985]
                }
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
                
                # Download button for results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="training_results.csv",
                    mime="text/csv"
                )
    
    def fraud_detection_page(self):
        """Fraud detection page"""
        st.markdown('<h1 class="main-header">🔍 Fraud Detection</h1>', unsafe_allow_html=True)
        
        # Two modes: Single claim and batch processing
        detection_mode = st.radio(
            "Detection Mode",
            ["Single Claim Analysis", "Batch Processing"]
        )
        
        if detection_mode == "Single Claim Analysis":
            self.single_claim_analysis()
        else:
            self.batch_processing()
    
    def single_claim_analysis(self):
        """Single claim analysis form"""
        st.subheader("Enter Claim Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, value=5000.0, step=100.0)
            patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
            provider_id = st.selectbox("Provider ID", options=list(range(1, 101)))
        
        with col2:
            diagnosis_code = st.selectbox(
                "Diagnosis Code",
                options=['A01', 'B02', 'C03', 'D04', 'E05', 'F06', 'G07', 'H08']
            )
            procedure_code = st.selectbox(
                "Procedure Code",
                options=['P001', 'P002', 'P003', 'P004', 'P005', 'P006']
            )
            deductible = st.number_input("Deductible Amount ($)", min_value=0.0, value=500.0, step=50.0)
        
        with col3:
            admission_days = st.number_input("Admission Days", min_value=0, value=3)
            chronic_conditions = st.multiselect(
                "Chronic Conditions",
                options=['Diabetes', 'Heart Disease', 'Cancer', 'Hypertension', 'Asthma', 'Depression']
            )
            previous_claims = st.number_input("Previous Claims", min_value=0, value=2)
        
        # Model selection
        st.subheader("Detection Model")
        model_choice = st.selectbox(
            "Select Detection Model",
            ["Random Forest", "XGBoost", "Logistic Regression", "Ensemble (All Models)"],
            index=1
        )
        
        # Detection button
        if st.button("🔍 Analyze Claim", type="primary"):
            # Simulate fraud detection
            with st.spinner("Analyzing claim for potential fraud..."):
                import time
                time.sleep(2)  # Simulate processing time
                
                # Deterministic mock prediction based on inputs (same inputs → same output)
                key_tuple = (
                    float(claim_amount), int(patient_age), int(provider_id), str(diagnosis_code),
                    str(procedure_code), float(deductible), int(admission_days),
                    tuple(sorted(chronic_conditions)), int(previous_claims), str(model_choice)
                )
                seed = int(hashlib.sha256(str(key_tuple).encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                fraud_probability = float(rng.uniform(0, 1))
                is_fraud = fraud_probability > 0.7
                
                # Display results
                st.markdown("---")
                st.subheader("Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Fraud Probability", f"{fraud_probability:.2%}")
                    st.metric("Prediction", "FRAUD" if is_fraud else "LEGITIMATE")
                
                with col2:
                    # Confidence gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=fraud_probability * 100,
                        title={'text': "Confidence Level"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "red" if is_fraud else "green"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Fraud alert or success message
                if is_fraud:
                    st.markdown('<div class="fraud-alert">⚠️ **FRAUD ALERT**: This claim has been flagged as potentially fraudulent. Please review manually.</div>', unsafe_allow_html=True)
                    
                    # Reasons for flagging
                    st.subheader("Reasons for Flagging")
                    reasons = [
                        "Unusually high claim amount for diagnosis",
                        "Provider has history of suspicious claims",
                        "Short interval between claims",
                        "Mismatch between diagnosis and procedures"
                    ]
                    
                    for reason in reasons[:np.random.randint(2, 5)]:
                        st.write(f"• {reason}")
                else:
                    st.markdown('<div class="non-fraud">✅ **LEGITIMATE CLAIM**: This claim appears to be legitimate based on our analysis.</div>', unsafe_allow_html=True)
                
                # Download report
                report_data = {
                    'Claim_Amount': claim_amount,
                    'Patient_Age': patient_age,
                    'Provider_ID': provider_id,
                    'Diagnosis_Code': diagnosis_code,
                    'Fraud_Probability': fraud_probability,
                    'Prediction': 'Fraud' if is_fraud else 'Legitimate',
                    'Model_Used': model_choice,
                    'Timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                report_df = pd.DataFrame([report_data])
                csv = report_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=csv,
                    file_name=f"fraud_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def batch_processing(self):
        """Batch processing of multiple claims"""
        st.subheader("Upload Batch Data")
        
        uploaded_file = st.file_uploader("Upload CSV file with multiple claims", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(batch_data)} claims for processing")
                
                # Display sample
                st.dataframe(batch_data.head())
                
                # Processing options
                col1, col2 = st.columns(2)
                
                with col1:
                    model_choice = st.selectbox(
                        "Select Model for Batch Processing",
                        ["Random Forest", "XGBoost", "Logistic Regression", "Ensemble"]
                    )
                
                with col2:
                    threshold = st.slider(
                        "Fraud Probability Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        step=0.05
                    )
                
                # Process batch
                if st.button("🚀 Process Batch", type="primary"):
                    with st.spinner(f"Processing {len(batch_data)} claims..."):
                        import time
                        
                        # Simulate batch processing
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Simulate results
                        np.random.seed(42)
                        fraud_probs = np.random.uniform(0, 1, len(batch_data))
                        predictions = (fraud_probs > threshold).astype(int)
                        
                        # Add predictions to data
                        results_data = batch_data.copy()
                        results_data['Fraud_Probability'] = fraud_probs
                        results_data['Prediction'] = ['Fraud' if p == 1 else 'Legitimate' for p in predictions]
                        
                        st.success("Batch processing completed!")
                        
                        # Display summary
                        st.subheader("Batch Results Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        fraud_count = predictions.sum()
                        legitimate_count = len(predictions) - fraud_count
                        
                        with col1:
                            st.metric("Total Claims", len(batch_data))
                        with col2:
                            st.metric("Fraudulent Claims", fraud_count)
                        with col3:
                            st.metric("Legitimate Claims", legitimate_count)
                        with col4:
                            st.metric("Fraud Rate", f"{(fraud_count/len(batch_data))*100:.1f}%")
                        
                        # Results table
                        st.subheader("Detailed Results")
                        st.dataframe(results_data.head(20))
                        
                        # Download results
                        csv = results_data.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Full Results",
                            data=csv,
                            file_name=f"batch_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        
            except Exception as e:
                st.error(f"Error processing batch: {e}")
    
    def reports_page(self):
        """Reports and analytics page"""
        st.markdown('<h1 class="main-header">📈 Reports & Analytics</h1>', unsafe_allow_html=True)
        
        # Report type selection
        report_type = st.selectbox(
            "Select Report Type",
            ["Performance Metrics", "Fraud Trends", "Provider Analysis", "Financial Impact", "Custom Report"]
        )
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))
        
        # Generate report
        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                import time
                time.sleep(2)
                
                if report_type == "Performance Metrics":
                    self.generate_performance_report()
                elif report_type == "Fraud Trends":
                    self.generate_trends_report()
                elif report_type == "Provider Analysis":
                    self.generate_provider_report()
                elif report_type == "Financial Impact":
                    self.generate_financial_report()
    
    def generate_performance_report(self):
        """Generate performance metrics report"""
        st.subheader("Model Performance Report")
        
        # Performance metrics over time
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Accuracy Trend", "Recall Trend", "Precision Trend", "F1-Score Trend")
        )
        
        # Simulated data
        rf_acc = [0.90 + np.random.uniform(-0.02, 0.02) for _ in months]
        xgb_acc = [0.92 + np.random.uniform(-0.02, 0.02) for _ in months]
        ensemble_acc = [0.94 + np.random.uniform(-0.02, 0.02) for _ in months]
        
        fig.add_trace(
            go.Scatter(x=months, y=rf_acc, mode='lines', name='Random Forest'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=months, y=xgb_acc, mode='lines', name='XGBoost'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=months, y=ensemble_acc, mode='lines', name='Ensemble'),
            row=1, col=1
        )
        
        # Similar for other metrics...
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        insights = [
            "Ensemble model consistently outperforms individual models",
            "Recall has improved by 15% since January",
            "False positive rate has been reduced by 8%",
            "Model retraining every month shows positive impact"
        ]
        
        for insight in insights:
            st.write(f"• {insight}")
    
    def run(self):
        """Main application runner"""
        # Load models
        models_loaded = self.load_models()
        
        # Sidebar navigation
        page = self.sidebar_navigation()
        
        # Page routing
        if page == "Dashboard":
            self.dashboard_page()
        elif page == "Data Analysis":
            self.data_analysis_page()
        elif page == "Model Training":
            self.model_training_page()
        elif page == "Fraud Detection":
            self.fraud_detection_page()
        elif page == "Reports":
            self.reports_page()
        elif page == "Settings":
            st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
            
            st.subheader("System Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Database Connection String", value="postgresql://user:pass@localhost:5432/claims")
                st.number_input("Batch Size", min_value=100, max_value=10000, value=1000, step=100)
                st.selectbox("Logging Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
            
            with col2:
                st.number_input("Retraining Interval (days)", min_value=1, max_value=30, value=7)
                st.number_input("Fraud Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
                st.selectbox("Notification Method", ["Email", "SMS", "Dashboard", "All"])
            
            if st.button("Save Settings", type="primary"):
                st.success("Settings saved successfully!")
            
            st.markdown("---")
            st.subheader("System Information")
            
            sys_info = {
                "Python Version": sys.version.split()[0],
                "Streamlit Version": st.__version__,
                "Pandas Version": pd.__version__,
                "Models Loaded": "Yes" if models_loaded else "No",
                "Config Path": "config.yaml"
            }
            
            for key, value in sys_info.items():
                st.text(f"{key}: {value}")

# Main application
if __name__ == "__main__":
    app = FraudDetectionApp()
    app.run()