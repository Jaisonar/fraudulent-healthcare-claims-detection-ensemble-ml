# Healthcare Fraud Detection System

## 📋 Project Overview
A comprehensive system for detecting fraudulent healthcare insurance claims using Ensemble Machine Learning and SMOTENC for handling class imbalance.

## 👥 Team Members
- Jai Sonar (Roll No. 56)
- Prekshit Sonawane (Roll No. 60)
- Tirtha Sonawane (Roll No. 61)
- Akansha Tingase (Roll No. 64)

# Fraudulent Healthcare Claims Detection (Ensemble ML)

![Python](https://img.shields.io/badge/python-3.9+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![GitHub Repo Size](https://img.shields.io/github/repo-size/Jaisonar/fraudulent-healthcare-claims-detection-ensemble-ml)
![Last Commit](https://img.shields.io/github/last-commit/Jaisonar/fraudulent-healthcare-claims-detection-ensemble-ml)

An end‑to‑end machine learning system for detecting fraudulent healthcare insurance claims using **ensemble models** and addressing **class imbalance** with techniques like SMOTENC.  
This repository includes data preprocessing, model training & evaluation, and an interactive dashboard for real-time analysis.

## 🚀 Project Overview

Healthcare fraud detection is a critical application of machine learning due to the increasing volume of fraudulent claims and their financial impact. This project uses ensemble machine learning models to identify potentially fraudulent claims with high accuracy.

🔍 **Highlights:**
- Handles imbalanced datasets using SMOTENC oversampling
- Trains and compares multiple ensemble models
- Provides performance evaluation and visual insights
- Includes a Streamlit dashboard for easy interaction

## 📌 Table of Contents

1. [Features](#-features)  
2. [Tech Stack](#-tech-stack)  
3. [Dataset](#-dataset)  
4. [Installation](#-installation)  
5. [Usage](#-usage)  
6. [Model Training & Evaluation](#-model-training--evaluation)  
7. [Dashboard](#-dashboard)  
8. [Project Screenshots](#-project-screenshots)  
9. [Results](#-results)  
10. [License](#-license)  
11. [Contact](#-contact)

## ⭐ Features

- 📊 Data preprocessing pipeline
- 🤖 Ensemble ML models: Random Forest, XGBoost, LightGBM
- 🛠️ Imbalanced data handling with SMOTENC
- 📈 Model evaluation and performance metrics
- 📍 Streamlit dashboard for real-time prediction

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| ML Libraries | scikit-learn, XGBoost, LightGBM, imbalanced-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Version Control | Git / GitHub |

## 📂 Dataset

The dataset used is from the **Kaggle Healthcare Provider Fraud Detection** competition and includes:

- `Train.csv` – Training data with labels  
- `Beneficiary.csv` – Patient demographics  
- `Inpatient.csv` – Inpatient claims  
- `Outpatient.csv` – Outpatient claims

*(Ensure you include the raw data or instructions to download if testing locally.)*

## 🧩 Installation

## step 1. Clone the repository  
```bash
git clone https://github.com/Jaisonar/fraudulent-healthcare-claims-detection-ensemble-ml.git
cd fraudulent-healthcare-claims-detection-ensemble-ml
```
## step 2. Create & activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```
## step 3. Install dependencies
```bash
pip install -r requirements.txt
```
## Usage 
Run the preprocessing and training pipeline
```bash
python run_pipeline.py
```
Run the simple prediction script
```bash
python run_simple.py
```
## 📊 Model Training & Evaluation

The training pipeline performs:

Data cleaning & preprocessing
SMOTENC sampling for class imbalance
Training multiple ensemble models
Performance evaluation with metrics such as accuracy, recall, precision
Results are saved in:
1. model_summary.csv

2. model_results.json

## 📈 Dashboard

Run the Streamlit dashboard:
```bash
streamlit run src/app.py
```
This visual dashboard offers:

1. Interactive fraud prediction
2. Model performance visualizations
3.Feature importance insights
