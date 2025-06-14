# Customer Churn Prediction - Telco Dataset 📉🔍

## 📌 Project Overview

This project aims to predict **customer churn** using a real-world Telco dataset. The objective is to help telecom companies **identify customers at risk of leaving**, allowing them to proactively take retention measures. This end-to-end machine learning pipeline demonstrates my ability to handle imbalanced classification problems with modern techniques like **SMOTE, hyperparameter tuning (GridSearchCV)**, and **ensemble modeling (VotingClassifier)**.

---

## 📊 Dataset Summary

- **Source**: IBM Sample Data – [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows**: 7,043
- **Target**: `Churn` (Yes/No → encoded as 1/0)
- **Features Used**:
  - Numerical: `tenure`, `MonthlyCharges`, `TotalCharges`
  - Categorical: `Contract`, `PaymentMethod`, `InternetService`, `OnlineSecurity`, `TechSupport`, `gender`, `PaperlessBilling`, `OnlineBackup`

---

## 🛠️ Installation

Clone the repo and install required dependencies:

```bash
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction
pip install -r requirements.txt
```
## 📁 Project Structure
```bash
telco-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── preprocess.py          # Preprocessing pipeline
├── train_model.py         # Model training and evaluation script
├── requirements.txt       # Required Python libraries
├── README.md              # Project documentation
```
## 🚀 Usage
To train the model and evaulate performance:
```bash
python train_model.py
```
This script:
- Preprocess data
- Handles class imbalance using `SMOTE`
- Tunes models using `GridSearchCV`
- Evaulates final predictions using a `VotingClassifier`
- Plots `ROC` and `Precision-Recall` curves

## 🧠 Models & Techniques
| Model            | Optimized With | Metric   |
| ---------------- | -------------- | -------- |
| RandomForest     | GridSearchCV   | F1-Score |
| LightGBM         | GridSearchCV   | F1-Score |
| XGBoost          | GridSearchCV   | F1-Score |
| GradientBoosting | GridSearchCV   | F1-Score |
| VotingClassifier | Soft Voting    | ROC-AUC  |

### 📌 Best Performance:
- *VotingClassifier* (Ensemble of best RF, LGBM, XGBoost, GradientBooster)
- *ROC AUC Score*: `0.81`
- *F1 Score(Churn Class)*: `0.58`
- *Classification Accuracy: `76%`

## 🎯 Skills Highlighted
✅ Data Cleaning & EDA
✅ Feature Engineering & Selection
✅ Imbalanced Learning with SMOTE
✅ Hyperparameter Tuning (GridSearchCV)
✅ Advanced ML Models: LightGBM, XGBoost, Ensembles
✅ ROC/PR Curve Interpretation
✅ Production-ready code structure

## 🔮 Future Improvements
- Model explainability with SHAP/LIME
- Flask or Streamlit API deployment
- Integration with live churn dashboard

## 👤 About Me
- I’m a data-driven problem solver with a passion for transforming raw data into actionable insights. This project reflects my proficiency in classification tasks, real-world deployment pipelines, and business impact-focused modeling.
- 🔗 [Linkedln](https://www.linkedin.com/in/alp-yaman-75a901174/) | 📧 alpyaman3@gmail.com




