# 🫀 Heart Disease Prediction System

A comprehensive machine learning-powered heart disease prediction system built with **Streamlit** and trained on the Cleveland Heart Disease dataset. This application compares multiple ML algorithms and deploys the best-performing model for accurate cardiovascular risk assessment.

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live_Demo-Visit_App-blue?style=for-the-badge)](https://aryanv2504-aryanv2504-heart-disease-prediction-strea-app-rvulie.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)](https://streamlit.io)


</div>

---

## 🚀 Live Application

**Try the app now:** [Heart Disease Prediction System](https://aryanv2504-aryanv2504-heart-disease-prediction-strea-app-rvulie.streamlit.app/)

---

## ✨ Key Features

- **🤖 Multi-Model Comparison**: Logistic Regression, Random Forest, and XGBoost evaluation
- **👥 Patient Management**: Complete patient registration and search functionality  
- **⚡ Risk Assessment Engine**: Clinical prediction using 7 key parameters
- **📊 Advanced Analytics**: ROC curves, confusion matrices, and feature importance analysis
- **🔄 Real-time Predictions**: Instant risk scoring with confidence intervals
- **🏥 Clinical Interpretability**: Clear risk factors and medical insights

---

## 📸 Application Screenshots

### Main Interface
![Screenshot 1](assets/ss%201.jpg)
![Screenshot 2](assets/ss%202.jpg)
![Screenshot 3](assets/ss%203.jpg)
![Screenshot 4](assets/ss%204.jpg)



---

## 📈 Model Performance & Analytics

### ROC Analysis
![ROC Curve](assets/SCREENSHOT%20-%20ROC%20CURVE.jpg)

### Feature Importance
![Feature Distribution](assets/SCREENSHOT-FEATURE%20DISTRIBUTION.jpg)

---

## 🔍 Confusion Matrix Results

### Logistic Regression (Selected Model)
![Logistic Regression Confusion Matrix](assets/SCREENSHOT-LOGISTIC%20REGRESSION-CONFUSION%20MATRIX.jpg)

### Random Forest
![Random Forest Confusion Matrix](assets/SCREENSHOT-RANDOM%20FOREST-CONFUSION%20MATRIX.jpg)
### XGBoost
![XGBoost Confusion Matrix](assets/SCREENSHOT-XG%20BOOST-CONFUSION%20MATRIX.jpg)

## 📋 Dataset Information

- **Source**: Cleveland Heart Disease Dataset
- **Features**: 13 clinical parameters including age, sex, chest pain type, blood pressure, cholesterol
- **Target Variable**: Binary classification (0: No Disease, 1: Disease Present)
- **Preprocessing**: Feature scaling, categorical encoding, and intelligent feature selection
- **Size**: 303 patient records with comprehensive medical data

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/aryanv2504/heart-disease-prediction-streamlit.git
cd heart-disease-prediction-streamlit
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the app**
Open your browser and navigate to [http://localhost:8501](http://localhost:8501)

---

## 📖 Usage Guide

### Step 1: 👤 Patient Registration
Enter patient demographic information and basic medical history through the intuitive form interface.

### Step 2: 🩺 Clinical Assessment
Input the following key parameters:
- **Age**: Patient age in years
- **Sex**: Male/Female
- **Chest Pain Type**: Typical angina, atypical angina, non-anginal pain, or asymptomatic
- **ST Depression**: Exercise-induced ST depression
- **Major Vessels**: Number of major vessels colored by fluoroscopy (0-3)
- **Thalassemia**: Blood disorder classification
- **Exercise Angina**: Exercise-induced angina (Yes/No)

### Step 3: 🎯 Generate Prediction
Click the prediction button to receive:
- **Risk Score**: Probability of heart disease (0-100%)
- **Confidence Level**: Model certainty in the prediction
- **Key Risk Factors**: Most influential clinical parameters
- **Medical Interpretation**: Clinical significance of the results

### Step 4: 📊 Review Analytics
Explore detailed model performance metrics, feature importance, and comparative analysis between different algorithms.

---

## Model Development Workflow

### Data Preprocessing
- Data cleaning and validation
- Categorical variable encoding
- Feature scaling and normalization
- Train-test-validation split (70-20-10)

### Model Training & Selection
- **Logistic Regression**: Selected as the primary model for optimal accuracy and interpretability
- **Random Forest**: Ensemble method for comparison
- **XGBoost**: Gradient boosting for performance benchmarking

### Model Evaluation
- Cross-validation with 5-fold strategy
- Performance metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion matrix analysis for each model
- Feature importance ranking

### Deployment
- Model serialization using pickle
- Integration with Streamlit interface
- Real-time prediction pipeline

---

## Project Structure

```
heart-disease-prediction-streamlit/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── app.py                             # Main Streamlit application
├── model/
│   ├── heart_disease_model.pkl        # Trained model (Logistic Regression)
│   └── model_training.ipynb           # Model development notebook
├── data/
│   └── cleveland_heart_disease.csv    # Training dataset
├── assets/
│   └── screenshots/                   # Application screenshots
│       ├── ss1.jpg                    # Main interface
│       ├── ss2.jpg                    # Patient registration
│       ├── ss3.jpg                    # Patient search
│       ├── ss4.jpg                    # Prediction results
│       ├── SCREENSHOT - ROC CURVE.jpg
│       ├── SCREENSHOT-BEST MODEL.jpg
│       ├── SCREENSHOT-FEATURE DISTRIBUTION.jpg
│       ├── SCREENSHOT-LOGISTIC REGRESSION-CONFUSION MATRIX.jpg
│       ├── SCREENSHOT-RANDOM FOREST-CONFUSION MATRIX.jpg
│       └── SCREENSHOT-XG BOOST-CONFUSION MATRIX.jpg
└── utils/
    ├── data_processing.py             # Data preprocessing utilities
    └── model_utils.py                 # Model helper functions
```

---

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Model Persistence**: pickle
- **Deployment**: Streamlit Cloud

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **85.2%** | **0.87** | **0.83** | **0.85** | **0.95** |
| Random Forest | 82.1% | 0.84 | 0.80 | 0.82 | 0.94 |
| XGBoost | 83.6% | 0.85 | 0.81 | 0.83 | 0.94 |

*Logistic Regression was selected as the primary model due to its superior AUC-ROC score of 0.95 and high interpretability in clinical settings.*

---

## 🤝 Contributing

We welcome contributions to improve the Heart Disease Prediction System:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes and test thoroughly**
4. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
5. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

---


## 👨‍💻 Developer

**Aryan Verma**  
- GitHub: [@aryanv2504](https://github.com/aryanv2504)
- Project Link: [Heart Disease Prediction System](https://github.com/aryanv2504/heart-disease-prediction-streamlit)
