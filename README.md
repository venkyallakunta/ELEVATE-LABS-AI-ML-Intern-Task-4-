# ELEVATE-LABS-AI-ML-Intern-Task-4-
# Step-1: Choose a binary classification dataset.
# Import libraries: import pandas as pd, import matplotlib.pyplot as plt, from sklearn.datasets import load_breast_cancer
# Load the dataset csv file i.e.,data=pd.read_csv(r'')
# print(x.head()) #It will print top 5 rows of the dataset
# Print(y.head()) # It will print bottom 5 rows of the dataset
# Step-2:Train/test split and standardize features
# Data Preparation: The dataset was divided into training and testing sets to evaluate model performance effectively.
# Feature Scaling: Numerical features were standardized using StandardScaler() to ensure all variables contribute equally to the model.
# Step-3: Fit a Logistic Regression model.
# Model Training: A Logistic Regression model was created using
# model = LogisticRegression(random_state=42)
# model.fit(x_train_scaled, y_train)
# Prediction:
# The model predicts probabilities for each class using: y_pred_prob = model.predict_proba(x_test_scaled)[:, 1]
# Step-4: Evaluate with confusion matrix, precision, reca l, ROC-AUC
# Confusion Matrix gives the count of correct and incorrect predictions:
# Term	Meaning
# TP	True Positive – correctly predicted positive class
# TN	True Negative – correctly predicted negative class
# FP	False Positive – predicted positive but actually negative
# FN	False Negative – predicted negative but actually positive
# Precision → How many predicted positives are actually positive.
# Precision=TP/TP + FP
# Recall (Sensitivity) → How many actual positives are correctly predicted.
# Recall= TP/TP+FN
# ROC-AUC → Tells how well the model separates the two classes.
# Closer the value to 1, the better the model performance.
# Step-5: Tune threshold and explain sigmoid function
# By default, Logistic Regression uses a 0.5 threshold.
# If predicted probability ≥ 0.5 → class 1
# If predicted probability < 0.5 → class 0
# Tuning the threshold helps adjust model behavior:
# Lower threshold → increases Recall (detects more positives)
# Higher threshold → increases Precision (reduces false positives)
# The sigmoid function converts any real number into a probability between 0 and 1.
# sigma(x)= 1/ 1+e-x.

