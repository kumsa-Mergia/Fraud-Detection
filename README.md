# 🛡️ Fraud Detection Project

This project aims to analyze and detect fraudulent activities using machine learning and explainable AI techniques. It covers both e-commerce fraud and credit card fraud scenarios. The pipeline includes data preprocessing, exploratory data analysis (EDA), model training with multiple algorithms, evaluation, and SHAP explainability.

---

## 📁 Project Structure

```

Fraud-Detection/
│
├── data/
│   ├── raw/                  # Original data (Fraud\_Data.csv, creditcard.csv, IpAddress\_to\_Country.csv)
│   ├── processed/            # Cleaned & engineered data
│   └── external/             # Third-party data (geolocation sources)
│
├── notebooks/
│   ├── eda/
│   │   ├── 01\_eda\_fraud.ipynb           # EDA on e-commerce fraud data
│   │   ├── 02\_eda\_credit.ipynb          # EDA on credit card data
│   │   ├── 03\_eda\_balanced\_fraud\_data.ipynb
│   ├── model\_training/
│   │   ├── creditcard\_model\_training.ipynb
│   │   ├── fraud\_model\_training.ipynb
│   └── 05\_shap\_analysis.ipynb           # SHAP explainability for best model
│
├── src/
│   ├── data\_preprocessing/
│   │   ├── **init**.py
│   │   ├── data\_preprocessor.py         # Pipeline for cleaning, transforming, and encoding data
│   ├── models/
│   │   ├── base\_model.py                # Abstract base class for all ML models
│   │   ├── logistic\_model.py            # Logistic Regression implementation
│   │   ├── xgboost\_model.py             # XGBoost Classifier
│   │   ├── ensemble\_model.py            # Random Forest Classifier
│   │   └── evaluator.py                 # Model evaluation and plotting
│   ├── explainability/
│   │   ├── **init**.py
│   │   ├── shap\_explainer.py            # SHAP value analysis
│
├── tests/
│   ├── test\_ip\_to\_int.py                # Unit tests for IP address to integer conversion
│   ├── test\_load\_data.py                # Data loading test cases
│   ├── test\_logistic\_model.py
│   ├── test\_xgboost\_model.py
│
├── requirements.txt                     # Required Python packages
├── .gitignore                           # Git ignore configuration
└── README.md                            # Project overview and setup

````

---

## 🧪 Features

- Detects **e-commerce fraud** and **credit card fraud**
- Cleans and preprocesses large datasets
- Builds and evaluates multiple ML models: Logistic Regression, XGBoost, Random Forest
- Performs **explainability analysis** using SHAP
- Fully modular pipeline with testable components

---

## ⚙️ Setup Instructions

1. Clone the repo:

```bash
git clone https://github.com/kumsa-Mergia/Fraud-Detection.git
cd Fraud-Detection
````

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Add your raw data to the `data/raw/` folder.

---

## 🚀 How to Run

1. Explore the data:

   * Run the EDA notebooks in `notebooks/eda/`.

2. Train models:

   * Open and run notebooks in `notebooks/model_training/`.

3. Evaluate explainability:

   * Run `05_shap_analysis.ipynb`.

4. Run tests:

   ```bash
   pytest tests/
   ```

---

## ✅ Tests

This project includes unit tests for data loading, IP conversion, and model logic. To run all tests:

```bash
pytest tests/
```

---

## 📊 SHAP Explainability

SHAP values are used to interpret model predictions and identify important features driving fraud detection decisions. This is essential for compliance and trust in high-stakes applications.

---

## 📌 Notes

* Raw datasets are not uploaded due to privacy and size. You must manually place them in `data/raw/`.
* Compatible with Python 3.8+

---

---

## 👨‍💻 Author

* **Kumsa Mergia** – *Infrastructure Engineer & ML Enthusiast*

---
