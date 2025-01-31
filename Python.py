#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:05:37 2025

@author: kiran
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import shap


# Load the dataset
file_path = "E Commerce Dataset.xlsx"
xls = pd.ExcelFile(file_path)

ecomm_data = pd.read_excel(xls, sheet_name='E Comm')

# Check for missing values
missing_values = ecomm_data.isnull().sum()

# Check for duplicate rows
duplicate_rows = ecomm_data.duplicated().sum()

# Drop duplicate rows if any
ecomm_data_cleaned = ecomm_data.drop_duplicates()

# Handling missing values

# Fill numerical columns with median
numerical_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
                  'OrderAmountHikeFromlastYear', 'CouponUsed', 
                  'OrderCount', 'DaySinceLastOrder']
for col in numerical_cols:
    ecomm_data_cleaned[col] = ecomm_data_cleaned[col].fillna(ecomm_data_cleaned[col].median())

# Verify missing values are handled
missing_values_after = ecomm_data_cleaned.isnull().sum()

#1. How does tenure impact churn rates?

plt.figure(figsize=(8,5))
sns.boxplot(x=ecomm_data_cleaned["Churn"], y=ecomm_data_cleaned["Tenure"], palette="coolwarm")
plt.title("Tenure Distribution by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Tenure (Months)")
plt.show()

# Calculate mean tenure for churned vs. retained customers
tenure_stats = ecomm_data_cleaned.groupby("Churn")["Tenure"].mean()
print("Mean Tenure for Churned and Retained Customers:\n", tenure_stats)

#2. Does customer satisfaction score correlate with churn?

plt.figure(figsize=(8,5))
sns.boxplot(x=ecomm_data_cleaned["Churn"], y=ecomm_data_cleaned["SatisfactionScore"], palette="magma")
plt.title("Satisfaction Score Distribution by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Satisfaction Score")
plt.show()

# Compute correlation
correlation = ecomm_data_cleaned[['SatisfactionScore', 'Churn']].corr()
print("Correlation between Satisfaction Score and Churn:\n", correlation)


#3. Do customers who raise complaints frequently have a higher churn rate?

plt.figure(figsize=(6,4))
sns.countplot(x=ecomm_data_cleaned["Complain"], hue=ecomm_data_cleaned["Churn"], palette="Set2")
plt.title("Churn Rate by Complaints")
plt.xlabel("Complained (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.show()

# Calculate churn rate for customers who complained vs. those who didn't
churn_by_complaint = ecomm_data_cleaned.groupby("Complain")["Churn"].mean()
print("Churn Rate by Complaint Status:\n", churn_by_complaint)

#4. How do order amounts change year-over-year for customers?

plt.figure(figsize=(8,5))
sns.boxplot(x=ecomm_data_cleaned["Churn"], y=ecomm_data_cleaned["OrderAmountHikeFromlastYear"], palette="coolwarm")
plt.title("Order Amount Hike from Last Year by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Order Amount Hike %")
plt.show()

# Calculate mean order amount hike for churned vs. retained customers
order_hike_stats = ecomm_data_cleaned.groupby("Churn")["OrderAmountHikeFromlastYear"].mean()
print("Mean Order Amount Hike by Churn:\n", order_hike_stats)

#5. Do customers who use coupons more frequently have a lower churn rate?

plt.figure(figsize=(8,5))
sns.boxplot(x=ecomm_data_cleaned["Churn"], y=ecomm_data_cleaned["CouponUsed"], palette="coolwarm")
plt.title("Coupon Usage by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Coupons Used")
plt.show()

# Calculate mean coupon usage for churned vs. retained customers
coupon_usage_stats = ecomm_data_cleaned.groupby("Churn")["CouponUsed"].mean()
print("Mean Coupon Usage by Churn:\n", coupon_usage_stats)


#6. Does the number of orders influence churn?

plt.figure(figsize=(8,5))
sns.boxplot(x=ecomm_data_cleaned["Churn"], y=ecomm_data_cleaned["OrderCount"], palette="coolwarm")
plt.title("Order Count by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Orders")
plt.show()

# Calculate mean order count for churned vs. retained customers
order_count_stats = ecomm_data_cleaned.groupby("Churn")["OrderCount"].mean()
print("Mean Order Count by Churn:\n", order_count_stats)


#7. What is the impact of cashback on customer retention?

plt.figure(figsize=(8,5))
sns.boxplot(x=ecomm_data_cleaned["Churn"], y=ecomm_data_cleaned["CashbackAmount"], palette="coolwarm")
plt.title("Cashback Amount by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Cashback Amount")
plt.show()

# Calculate mean cashback for churned vs. retained customers
cashback_stats = ecomm_data_cleaned.groupby("Churn")["CashbackAmount"].mean()
print("Mean Cashback Amount by Churn:\n", cashback_stats)


#8. Which customer segments (city tier, marital status, gender) are more likely to churn?

plt.figure(figsize=(8,5))
gender_churn_counts = ecomm_data_cleaned.groupby("Gender")["Churn"].value_counts().unstack()
gender_churn_counts.plot(kind='pie', subplots=True, autopct='%1.1f%%', figsize=(10,6), legend=True, labels=["Male - No Churn", "Male - Churn", "Female - No Churn", "Female - Churn"])
plt.title("Churn Distribution by Gender")
plt.ylabel("")
plt.show()

#9. How does the preferred login device impact customer behavior and retention?

plt.figure(figsize=(8,5))
sns.countplot(x=ecomm_data_cleaned["PreferredLoginDevice"], hue=ecomm_data_cleaned["Churn"], palette="Set1")
plt.title("Churn Rate by Preferred Login Device")
plt.xlabel("Preferred Login Device")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Churn", labels=["No", "Yes"])
plt.show()


#10. Which preferred payment mode is associated with the highest customer retention?

plt.figure(figsize=(8,5))
sns.countplot(x=ecomm_data_cleaned["PreferredPaymentMode"], hue=ecomm_data_cleaned["Churn"], palette="Set2")
plt.title("Churn Rate by Preferred Payment Mode")
plt.xlabel("Preferred Payment Mode")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Churn", labels=["No", "Yes"])
plt.show()


#11. Do customers with multiple registered devices tend to stay longer?

plt.figure(figsize=(8,5))
sns.boxplot(x=ecomm_data_cleaned["Churn"], y=ecomm_data_cleaned["NumberOfDeviceRegistered"], palette="coolwarm")
plt.title("Number of Registered Devices by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Devices")
plt.show()


#12. Is there a relationship between hours spent on the app and churn?

plt.figure(figsize=(8,5))
sns.boxplot(x=ecomm_data_cleaned["Churn"], y=ecomm_data_cleaned["HourSpendOnApp"], palette="coolwarm")
plt.title("Hours Spent on App by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Hours Spent on App")
plt.show()


#13. Does order frequency decrease before a customer churns?

plt.figure(figsize=(8,5))
sns.boxplot(x=ecomm_data_cleaned["Churn"], y=ecomm_data_cleaned["OrderCount"], palette="coolwarm")
plt.title("Order Count by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Order Count")
plt.show()

#14. How does the time since the last order affect the likelihood of churn?

plt.figure(figsize=(8,5))
sns.boxplot(x=ecomm_data_cleaned["Churn"], y=ecomm_data_cleaned["DaySinceLastOrder"], palette="coolwarm")
plt.title("Days Since Last Order by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Days Since Last Order")
plt.show()


##########################################################################

file_path = "E Commerce Dataset.xlsx"
xls = pd.ExcelFile(file_path)

ecomm_data = pd.read_excel(xls, sheet_name='E Comm')

# Check for missing values
missing_values = ecomm_data.isnull().sum()

# Check for duplicate rows
duplicate_rows = ecomm_data.duplicated().sum()

# Drop duplicate rows if any
ecomm_data_cleaned = ecomm_data.drop_duplicates()

# Handling missing values

# Fill numerical columns with median
numerical_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
                  'OrderAmountHikeFromlastYear', 'CouponUsed', 
                  'OrderCount', 'DaySinceLastOrder']
for col in numerical_cols:
    ecomm_data_cleaned[col] = ecomm_data_cleaned[col].fillna(ecomm_data_cleaned[col].median())

# Encode categorical variables
categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'MaritalStatus', 'Gender', 'PreferedOrderCat']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    ecomm_data_cleaned[col] = le.fit_transform(ecomm_data_cleaned[col])
    label_encoders[col] = le

# Define features and target variable
X = ecomm_data_cleaned.drop(columns=['Churn', 'CustomerID'])  # Dropping target and unique ID
y = ecomm_data_cleaned['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize numerical features
scaler = StandardScaler()
X_train_resampled[numerical_cols] = scaler.fit_transform(X_train_resampled[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Data preprocessing complete

# Train Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print Model Performance Metrics
print("Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Logistic Regression Model complete


# Train Decision Tree Model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred_tree = decision_tree.predict(X_test)
y_prob_tree = decision_tree.predict_proba(X_test)[:, 1]

# Model Evaluation
accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree)
recall_tree = recall_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)
roc_auc_tree = roc_auc_score(y_test, y_prob_tree)
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)

# Print Model Performance Metrics
print("Decision Tree Model Performance:")
print(f"Accuracy: {accuracy_tree:.4f}")
print(f"Precision: {precision_tree:.4f}")
print(f"Recall: {recall_tree:.4f}")
print(f"F1-score: {f1_tree:.4f}")
print(f"ROC AUC Score: {roc_auc_tree:.4f}")
print("Confusion Matrix:")
print(conf_matrix_tree)

# Decision Tree Model complete


# Hyperparameter tuning for Decision Tree
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model from GridSearchCV
best_decision_tree = grid_search.best_estimator_

# Train optimized Decision Tree Model
best_decision_tree.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred_tree = best_decision_tree.predict(X_test)
y_prob_tree = best_decision_tree.predict_proba(X_test)[:, 1]

# Model Evaluation
accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree)
recall_tree = recall_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)
roc_auc_tree = roc_auc_score(y_test, y_prob_tree)
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)

# Print Optimized Model Performance Metrics
print("Optimized Decision Tree Model Performance:")
print(f"Accuracy: {accuracy_tree:.4f}")
print(f"Precision: {precision_tree:.4f}")
print(f"Recall: {recall_tree:.4f}")
print(f"F1-score: {f1_tree:.4f}")
print(f"ROC AUC Score: {roc_auc_tree:.4f}")
print("Confusion Matrix:")
print(conf_matrix_tree)

# Decision Tree Model with Hyperparameter Tuning Complete


# Feature Importance Analysis
feature_importances = best_decision_tree.feature_importances_
features = X.columns

# Plot Feature Importance
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances, y=features, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance from Optimized Decision Tree")
plt.show()

# Decision Tree Model with Hyperparameter Tuning and Feature Importance Analysis Complete

