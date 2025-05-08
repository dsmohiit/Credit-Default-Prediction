# 💳 Credit Default Prediction Using Logistic Regression and Random Forest

This repository presents a machine learning pipeline to predict credit card default using customer demographic and payment history data. The project is built on a cleaned and structured dataset inspired by a real-world financial use case.

## 📘 Project Overview

This notebook includes:

* Exploratory Data Analysis (EDA)
* Data Preprocessing & Cleaning
* Feature Engineering
* Feature Selection using `SelectKBest`
* Model Training:

  * Logistic Regression
  * Random Forest Classifier
* Evaluation using classification reports, confusion matrices, and accuracy scores

## 📁 Dataset

The dataset includes information about clients' credit limits, repayment status, bill amounts, payments, and default labels. Key features:

* `LIMIT_BAL`: Credit limit
* `PAY_0` to `PAY_6`: Repayment history
* `BILL_AMT1` to `BILL_AMT6`: Monthly bill amounts
* `PAY_AMT1` to `PAY_AMT6`: Monthly payments
* `default.payment.next.month`: Target label (1 = Default, 0 = No Default)

## 🧪 Steps Performed

### 1. EDA

* Distribution plots
* Count plots by category (e.g., default rate by gender)

### 2. Data Cleaning

* Handled missing values via imputation
* Removed irrelevant features (`ID`)

### 3. Feature Selection

* `SelectKBest` with `chi2` to extract top predictors

### 4. Modeling

* Used `LogisticRegression` and `RandomForestClassifier`
* Evaluated with accuracy, precision, recall, and F1-score

## 📊 Results

* Logistic Regression: Baseline model for interpretability
* Random Forest: Stronger performance with nonlinear decision boundaries

## ⚙️ Requirements

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install via:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Notebook

Open in Google Colab:
[Run on Colab](https://colab.research.google.com/drive/1vsUkFP1DjpPadD9nZn1MhqPMejhrMFej)

## 📌 Notes

* Missing values were initially handled using `SimpleImputer`.
* Class imbalance (if any) can be addressed using oversampling/undersampling (not covered here).
* You can expand this work with XGBoost or hyperparameter tuning.
