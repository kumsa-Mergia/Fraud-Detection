# 🛡️ Fraud Detection Project

This project aims to detect fraudulent transactions using **machine learning** and **explainable AI**. It supports both **e-commerce fraud detection** and **credit card fraud detection** using a modular and test-driven workflow.

The pipeline covers:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Model training and evaluation
* Explainability using SHAP

---

## 📁 Project Structure

```bash
Fraud-Detection/
│
├── data/
│   ├── raw/                        # Raw datasets (Fraud_Data.csv, creditcard.csv, IpAddress_to_Country.csv)
│   ├── processed/                  # Cleaned and feature-engineered datasets
│
├── notebooks/
│   ├── eda/                        # EDA notebooks
│   │   ├── 01_eda_fraud.ipynb
│   │   ├── 02_eda_credit.ipynb
│   │   ├── 03_eda_balanced_fraud_data.ipynb
│   ├── model_training/            # Model training notebooks
│   │   ├── creditcard_model_training.ipynb
│   │   ├── fraud_model_training.ipynb
│   │   ├── balanced_fraud_data_model_training.ipynb
│   │   └── README.md
│   ├── explainability/            # SHAP and model interpretability
│   │   └──XGBoost_explainability.ipynb
│   │   └──RandomForest_explainability.ipynb.ipynb
│   │   └── README.md
│   ├── data_processing.ipynb
│   └── README.md
│
├── src/
│   ├── data_preprocessing/
│   │   ├── __init__.py
│   │   └── data_preprocessor.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── logistic_model.py
│   │   ├── xgboost_model.py
│   │   ├── ensemble_model.py
│   │   └── evaluator.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_explainer.py
│
├── tests/
│   ├── test_ip_to_int.py
│   ├── test_load_data.py
│   ├── test_logistic_model.py
│   ├── test_xgboost_model.py
│
├── requirements.txt               # Python package dependencies
├── README.md                      # This file
└── .gitignore                     # Files and folders to exclude from Git
```

---

## 🧪 Features

* 🧹 Clean and preprocess raw data from multiple sources
* 📊 Perform in-depth **EDA** for insight extraction
* 🧠 Train & evaluate multiple ML models:

  * Logistic Regression
  * XGBoost
  * Random Forest
* 🎯 Evaluate models using F1 Score, AUC-PR, and Confusion Matrix
* 🔍 Use **SHAP** for interpretability and explainability
* 🧪 Includes unit tests for critical components
* 📦 Modular structure (ideal for CI/CD and production deployment)

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/kumsa-Mergia/Fraud-Detection.git
cd Fraud-Detection
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add raw data**

Place raw datasets in the `data/raw/` directory:

* `Fraud_Data.csv`
* `creditcard.csv`
* `IpAddress_to_Country.csv`

---

## 🚀 How to Use

### 📊 1. Explore the Data

Run the EDA notebooks in:

```bash
notebooks/eda/
```

### 🧠 2. Train Models

Open and run the training notebooks:

```bash
notebooks/model_training/
```

Includes:

* Evaluation metrics (F1, AUC-PR)
* Confusion matrices
* Model comparison

### 🔍 3. Explain Predictions

Run SHAP-based explainability in:

```bash
notebooks/explainability/model_explainability.ipynb
```

### ✅ 4. Run Unit Tests

```bash
pytest tests/
```

---

## 🧠 Model Performance Highlights

| Dataset     | Model         | AUC-PR | F1 Score | Accuracy |
| ----------- | ------------- | ------ | -------- | -------- |
| Credit Card | Logistic      | 0.6877 | 0.0996   | 97.0%    |
|             | XGBoost       | 0.8036 | 0.8333   | 99.9%    |
|             | Random Forest | 0.8196 | 0.8178   | 99.9%    |
| Fraud       | Logistic      | 0.4522 | 0.2682   | 64.0%    |
|             | XGBoost       | 0.5967 | 0.6699   | 95.0%    |
|             | Random Forest | 0.6208 | 0.6951   | 96.0%    |

---

## 📊 SHAP Explainability

This project uses SHAP (SHapley Additive exPlanations) to visualize:

* Global feature importance
* Local instance-based explanations
* Feature effects on predictions

---

## 📌 Notes

* 📂 Raw datasets are excluded due to privacy and size constraints.
* 🐍 Python 3.8+ is recommended.
* ☑️ Project follows a modular and testable architecture for scalability and reproducibility.

---

## 👨‍💻 Author

**Kumsa Mergia**
📫 [kumsa-mergia.github.io](https://github.com/kumsa-Mergia)

---