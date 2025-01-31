# Churn Prediction Using Machine Learning

## Overview
This repository contains an end-to-end churn prediction analysis using machine learning models on an **E-Commerce Dataset**. The project includes **EDA, feature engineering, model training, hyperparameter tuning, and model interpretability using SHAP**.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [Feature Importance Analysis](#feature-importance-analysis)
- [SHAP Interpretability](#shap-interpretability)
- [Conclusion](#conclusion)
- [Repository Structure](#repository-structure)
- [How to Run the Project](#how-to-run-the-project)

---

# Introduction
Customer churn is a significant challenge for e-commerce businesses. This project aims to **predict customer churn** using various machine learning techniques and derive actionable insights to reduce churn rates.

---

# Dataset Overview
The dataset consists of various customer-related features, including:
- **Demographics**: Gender, Marital Status, City Tier
- **Behavioral Metrics**: Number of orders, Complaints, Satisfaction Score
- **Financial Metrics**: Cashback Amount, Coupon Usage, Order Amount Hike from Last Year

**Target Variable:** `Churn` (1 = Churned, 0 = Retained)

---

# Exploratory Data Analysis (EDA)
## Churn Distribution
![image](https://github.com/user-attachments/assets/93622fb5-de3b-4396-8bec-376f557736b7)

- Customers who raised complaints tend to churn more.
- Addressing customer complaints effectively can help reduce churn.

- The dataset is **imbalanced**, with fewer churned customers than retained ones.
- We used **SMOTE** to balance the classes before training the models.

## Feature Correlation Heatmap
![image](https://github.com/user-attachments/assets/e77aae56-7156-4190-bad4-7f15bf4013be)

- Males have a slightly higher churn rate than females.
- Understanding gender-based behavior can help tailor retention strategies.

- **Tenure and Cashback Amount** show a strong relationship with churn.
- Some categorical variables are converted using **Label Encoding**.

## Boxplots for Numerical Features
![image](https://github.com/user-attachments/assets/3a0e79d5-d1a6-4acb-a9ab-0fb6cf85db4f)

- Customers with lower tenure are more likely to churn.
- Long-term customers have a significantly lower churn rate.

![image](https://github.com/user-attachments/assets/0b881f0c-7c77-47f1-8744-0cdd3fedcd4c)

- Churned customers tend to have a lower satisfaction score.
- Higher satisfaction scores are associated with customer retention.

![image](https://github.com/user-attachments/assets/317f5e89-816a-49f6-b252-ad360e8e80e3)

- Customers who churn tend to receive lower cashback amounts.
- Higher cashback incentives are associated with lower churn rates.

## Categorical Feature Analysis
![image](https://github.com/user-attachments/assets/8922ef26-181f-45c9-998e-7435964623d4)

- Customers from **City Tier 1** have the highest retention rates.
- **City Tier 2 customers** have the highest churn rate.

![image](https://github.com/user-attachments/assets/ad2cbe91-d3b1-4402-97d2-b72fa5c74638)

- **Single customers** have a higher churn rate compared to married and divorced customers.
- Retention strategies can be tailored based on marital status insights.

![image](https://github.com/user-attachments/assets/3c98fb21-a3c8-41f5-8bda-c1b14b7884e8)

- Customers who prefer using a **mobile phone** to log in churn more frequently.
- Desktop users have a higher retention rate.

![image](https://github.com/user-attachments/assets/d2743a58-f17f-446c-bfd2-ed1280615d23)

- Customers who use **credit cards and debit cards** have the highest retention rates.
- Those relying on **cash on delivery and e-wallets** are more likely to churn.

## Behavioral Insights
![image](https://github.com/user-attachments/assets/c6e8a4b3-cab7-4792-b55d-b463943cb883)

- Customers who churn tend to spend less time on the app.
- Higher app engagement is correlated with lower churn rates.

![image](https://github.com/user-attachments/assets/1da8ee2c-24e0-489a-852d-9a0d812b3206)

- Customers who haven't ordered recently are more likely to churn.
- Encouraging frequent purchases can help retain customers.

---

# Data Preprocessing
## Key Steps:
- **Handling Missing Values**: Imputation using **median** for numerical columns.
- **Encoding Categorical Variables**: Using **Label Encoding**.
- **Scaling**: Applied **StandardScaler** for numerical features.
- **Class Balancing**: Used **SMOTE** to address class imbalance.

---

# Model Training & Evaluation
## Logistic Regression
**[Insert Logistic Regression Performance Metrics Here]**

## Decision Tree (Optimized)
**[Insert Decision Tree Performance Metrics Here]**

- After hyperparameter tuning, **Decision Tree** performed significantly better than Logistic Regression.

---

# Feature Importance Analysis
## Decision Tree Feature Importance
![image](https://github.com/user-attachments/assets/b850e8f6-e928-45a9-8249-9bd75b8dc8d9)

- **Tenure** is the most significant predictor of churn.
- **Cashback Amount and Days Since Last Order** also strongly impact retention.
- **Complaints and Order Count** contribute to churn prediction.

---

# SHAP Interpretability
## SHAP Summary Plot
![image](https://github.com/user-attachments/assets/d04e7ad3-b95c-4363-a98a-1ad39df36a3b)

- SHAP values help explain how each feature contributes to the model’s predictions.
- Higher **Tenure** and **Cashback Amount** reduce churn likelihood.

---

# Conclusion
- **Longer Tenure and Higher Cashback Amount** significantly reduce churn.
- **Customers with high Order Amount Hike tend to churn** more.
- The **Decision Tree Model** with hyperparameter tuning provides a **strong predictive capability**.
- **SHAP Analysis** enhances interpretability, making the model insights actionable.

## Next Steps
- Deploy the model as a **web API**.
- Further **optimize hyperparameters** using **RandomizedSearchCV**.
- Explore **deep learning models** for more advanced predictions.

---

# Repository Structure
```
📂 Churn_Prediction
│-- 📄 README.md
│-- 📄 churn_analysis.ipynb  # Jupyter Notebook with Code
│-- 📂 data  # Contains dataset
│-- 📂 images  # Contains graphs
│-- 📂 models  # Saved models for inference
```

---

# 🚀 How to Run the Project
```bash
pip install -r requirements.txt
python churn_analysis.py
```

**Author**: [Kiran Ranganali]

