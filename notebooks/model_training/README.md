# 📂 Model Training Notebooks

This directory contains Jupyter notebooks for training and evaluating machine learning models on both **credit card transaction data** and **e-commerce fraud data**. The goal is to build classification models that accurately detect fraudulent transactions.

---

## 📄 Contents

* `creditcard_model_training.ipynb`
  → Trains and evaluates models (Logistic Regression, XGBoost, Random Forest) on credit card transaction data.

* `fraud_model_training.ipynb`
  → Trains and evaluates the same set of models on e-commerce fraud detection data.

---

## 🧠 Models Trained

| Dataset         | Model              | AUC-PR | F1 Score | Accuracy | Notes                                                       |
| --------------- | ------------------ | ------ | -------- | -------- | ----------------------------------------------------------- |
| **Credit Card** | LogisticRegression | 0.6877 | 0.0996   | 0.97     | High precision for class 0, poor recall for fraud (class 1) |
|                 | XGBoost            | 0.8036 | 0.8333   | \~1.00   | Strong fraud detection with balanced performance            |
|                 | RandomForest       | 0.8196 | 0.8178   | \~1.00   | Slightly better than XGBoost in AUC-PR                      |
| **Fraud**       | LogisticRegression | 0.4522 | 0.2682   | 0.64     | Weak fraud recall; model underfits minority class           |
|                 | XGBoost            | 0.5967 | 0.6699   | 0.95     | Best recall balance among all                               |
|                 | RandomForest       | 0.6208 | 0.6951   | 0.96     | Best overall performance on fraud data                      |

---
##   Model for Balanced fraud dataset:

The evaluation results clearly show that **XGBoost** significantly outperforms **Logistic Regression** across all major metrics.

| **Metric**            | **XGBoost** | **Logistic Regression** |
|------------------------|-------------|--------------------------|
| AUC-PR                 | **0.9780**  | 0.8087                   |
| F1 Score               | **0.9439**  | 0.6835                   |
| Accuracy               | **~95%**    | ~67%                     |
| Recall (Fraud Class)   | **0.90**    | 0.70                     |
| Precision (Fraud Class)| **1.00**    | 0.66                     |




## 📝 Classification Reports Summary

### 📌 Credit Card Dataset

#### Logistic Regression

* **F1 Score:** 0.0996
* **Precision (fraud):** 0.05
* **Recall (fraud):** 0.89

#### XGBoost

* **F1 Score:** 0.8333
* **Precision (fraud):** 0.90
* **Recall (fraud):** 0.77

#### Random Forest

* **F1 Score:** 0.8178
* **Precision (fraud):** 0.96
* **Recall (fraud):** 0.71

---

### 📌 Fraud Dataset

#### Logistic Regression

* **F1 Score:** 0.2682
* **Precision (fraud):** 0.17
* **Recall (fraud):** 0.70

#### XGBoost

* **F1 Score:** 0.6699
* **Precision (fraud):** 0.89
* **Recall (fraud):** 0.54

#### Random Forest

* **F1 Score:** 0.6951
* **Precision (fraud):** 1.00
* **Recall (fraud):** 0.53

---

## ✅ Key Takeaways

* Tree-based models (XGBoost and Random Forest) significantly outperform Logistic Regression in detecting fraud.
* Random Forest shows a slight edge in performance (especially AUC-PR) across both datasets.
* Logistic Regression struggles with class imbalance and underperforms on fraud detection tasks.
* Evaluation metrics like **AUC-PR**, **F1 Score**, and **Confusion Matrix** are essential due to the imbalanced nature of the datasets.

---

## 🚀 How to Run

1. Navigate to the root of the project.
2. Open the Jupyter environment:

```bash
jupyter lab notebooks/model_training/
```

3. Run `creditcard_model_training.ipynb` or `fraud_model_training.ipynb` to see model training and evaluation results.

---

## 📎 Dependencies

Make sure the following packages are installed:

```bash
scikit-learn
xgboost
imbalanced-learn
numpy
pandas
matplotlib
```

Check the `requirements.txt` for a complete list.
