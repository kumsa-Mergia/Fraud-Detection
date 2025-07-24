# 📓 Notebooks Directory

This directory contains organized Jupyter notebooks related to the **E-commerce Fraud Detection Project**, grouped by their respective purposes: exploratory data analysis (EDA), model training, and model explainability.

## 📁 Structure Overview

```
notebooks/
├── eda/
│   ├── 01_eda_fraud.ipynb
│   ├── 02_eda_credit.ipynb
│   └── 03_eda_balanced_fraud_data.ipynb
├── model_training/
│   ├── creditcard_model_training.ipynb
│   └── fraud_model_training.ipynb
├── explainability/
│   └── model_explainability.ipynb
└── data_processing.ipynb
```

---

## 📊 EDA Notebooks (`notebooks/eda/`)

These notebooks provide an in-depth exploratory data analysis of the raw and preprocessed datasets:

* **`01_eda_fraud.ipynb`**: EDA on the e-commerce fraud transactions dataset.
* **`02_eda_credit.ipynb`**: EDA on credit card transaction data.
* **`03_eda_balanced_fraud_data.ipynb`**: EDA on the balanced version of the fraud dataset after handling class imbalance.

---

## 🧠 Model Training (`notebooks/model_training/`)

These notebooks include model development, training, evaluation, and performance analysis:

* **`creditcard_model_training.ipynb`**: Trains and evaluates models using the credit card dataset.
* **`fraud_model_training.ipynb`**: Trains and evaluates models on the fraud detection dataset.

---

## 🔍 Explainability (`notebooks/explainability/`)

* **`model_explainability.ipynb`**: Contains visualizations and interpretability analysis (e.g., SHAP, LIME) for trained fraud detection models.

---

## ⚙️ Data Preprocessing

* **`data_processing.ipynb`**: Handles initial data cleaning, feature engineering, and transformation steps before EDA and model training.

---

## 💡 Usage

1. Clone the repository and navigate to the project root.
2. Ensure dependencies are installed (see `requirements.txt`).
3. Open Jupyter Lab or Jupyter Notebook and navigate to the desired notebook.

```bash
jupyter lab notebooks/
```

---

## 📌 Notes

* The project uses synthetic and/or anonymized financial transaction data for fraud detection experimentation.
* Model training leverages Scikit-learn, Imbalanced-learn, and other standard ML libraries.
* Explainability includes SHAP or LIME to interpret model predictions and understand feature impact.

---

