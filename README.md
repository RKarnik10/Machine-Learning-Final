# Titanic Survival Prediction — ML Classification Project

Final project for **CIS053 (Intro to Machine Learning)** at Mission College, taught by Prof. Jahan Ghofraniha. Predicts passenger survival on the Titanic using the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic) by training and comparing seven classification models.

**Authors:** Rehaan Karnik, Soham Kulkarni

---

## Overview

Binary classification task: given a passenger's features (class, sex, age, family aboard, ticket, fare, cabin), predict whether they survived. The goal is a model that balances bias and variance — targeting the ~80% accuracy threshold that top public Kaggle solutions tend to converge on.

## Data Preparation

- Dropped unused columns (indices 0 and 11)
- Converted string columns (`Sex`, `Ticket`, `Fare`, `Cabin`) to integer encodings so they could be fed into the models
- Trained on `training.csv`, predicted on `testing.csv`

## EDA

Ran three passes on the data — **standard**, **normalized**, and **standardized** — producing descriptive stats, histograms, scatterplot matrices, and Pearson correlation heatmaps for each. **Normalized data** was chosen for modeling because it cleaned the distributions most effectively.

Notable correlations in the raw data:
- `Fare` ↔ `Pclass`: **−0.75**
- `Cabin` ↔ `Pclass`: **+0.69**
- `Fare` ↔ `Cabin`: **−0.57**

## Models Compared

All seven models trained on normalized training data and evaluated with cross-validation and ROC curves:

| Model | Best Accuracy | CV Mean | CV Std |
|---|---|---|---|
| **Random Forest** | **82.5%** | **81.2%** | 0.038 |
| Bagging | 82.5% | 79.1% | 0.043 |
| AdaBoost | 81.8% | 80.1% | 0.032 |
| Standard Decision Tree | 81.1% | 78.9% | 0.037 |
| MLP Classifier | 72.7% | 68.9% | 0.034 |
| Kernel SVC | 71.3% | 66.7% | 0.031 |
| Linear SVC | 70.6% | 67.9% | 0.032 |

## ROC AUC Scores

| Model | AUC |
|---|---|
| **Random Forest** | **0.91** |
| Bagging | 0.91 |
| AdaBoost | 0.88 |
| MLP Classifier | 0.80 |
| Standard Decision Tree | 0.80 |

## Conclusion

**Random Forest was selected as the final model.** It tied Bagging for top accuracy (82.5%) and top ROC AUC (0.91), but edged out Bagging on cross-validation mean accuracy — indicating slightly better generalization. The resulting ~81% accuracy hits the bias-variance sweet spot observed across top public Kaggle solutions.

Applied to `testing.csv`, the predictions are saved in `passenger_prediction.csv`. **Note:** the model predicted more deaths than the actual test-set ground truth reflects.

## Files

- `Machine_Learning_Final.ipynb` — Jupyter notebook with full EDA, model training, and predictions
- `Machine_Learning_Final.pdf` — Exported notebook as PDF
- `passenger_prediction.csv` — Final predictions from the Random Forest model
