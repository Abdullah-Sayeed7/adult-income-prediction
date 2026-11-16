# Adult Income Prediction API

## Problem Description

The goal of this project is to predict whether an individual's annual income exceeds $50,000 based on demographic and employment data (Census Income dataset). This is a binary classification problem with a significant class imbalance (approx. 76% <=50K, 24% >50K). Accurate predictions can be used by financial institutions for targeted marketing or credit risk assessment.

## Dataset

- **Source:** UCI Machine Learning Repository (Adult Income Dataset).
- **Data:** Committed in the repo at `data/adult.data`.
- **Features:** 14 features including age, workclass, education, marital status, occupation, and hours worked per week.

## Exploratory Data Analysis (EDA)

- Handled missing values encoded as `?` in `workclass`, `occupation`, and `native-country`.
- Observed that `education-num` is highly correlated with income; those with education levels 15-16 (Doctorate/Prof-school) have a >70% chance of earning >50K.
- Identified massive zero-inflation in `capital-gain` and `capital-loss`.
- Dropped `fnlwgt` (census sampling weight) as it does not represent individual characteristics.

## Models Evaluated

We trained and compared three models using Scikit-Learn Pipelines:

1. **Logistic Regression** (Baseline): Test ROC-AUC: ~0.907
2. **Decision Tree** (max_depth=10): Test ROC-AUC: ~0.906
3. **Random Forest** (Tuned via GridSearchCV): Test ROC-AUC: **0.919** (Best Model)

**Final Model:** A Tuned Random Forest Classifier (`max_depth=20`, `min_samples_leaf=5`). It generalized best, resolving the overfitting observed in the untuned Random Forest, achieving a test accuracy of 86.7% and an F1-score of 0.68.

## Project Structure

```text
├── data/adult.data         # Dataset
├── notebook.ipynb          # EDA, Model Selection, and Tuning
├── train.py                # Script to train the model and save model.bin
├── predict.py              # FastAPI service for predictions
├── model.bin               # Saved Scikit-Learn pipeline
├── requirements.txt        # Dependencies
├── Dockerfile              # Container setup
└── README.md
```

## Cloud Deployment

**Live API URL:** https://adult-income-api-kz1t.onrender.com
**Testing:** You can send a POST request to the `/predict` endpoint of this URL.
