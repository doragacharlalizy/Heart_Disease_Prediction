# Heart Disease Prediction Project

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
    1. [Data Loading](#data-loading)
    2. [Data Cleaning](#data-cleaning)
    3. [Data Conversion](#data-conversion)
    4. [Data Trimming](#data-trimming)
3. [Data Transformation](#data-transformation)
4. [Model Building and Evaluation](#model-building-and-evaluation)
    1. [Models Used](#models-used)
    2. [Model Training](#model-training)
    3. [Model Prediction](#model-prediction)
    4. [Model Evaluation](#model-evaluation)
5. [Results](#results)
    1. [Model Performance](#model-performance)
    2. [Best Parameters (Random Forest)](#best-parameters-random-forest)
    3. [Feature Importance](#feature-importance)
6. [Conclusion](#conclusion)

## 1. Introduction
This document provides a comprehensive overview of the heart disease prediction project, covering data preparation, transformation, model building, and evaluation. The goal of the project is to predict heart disease using various machine learning models.

## 2. Data Preparation

### 2.1 Data Loading
The data is loaded from a CSV file:
python
import pandas as pd
data = pd.read_csv('C:\\Users\\samee\\PycharmProjects\\Heart_Disease\\Heart_Disease_Prediction.csv')

### 2.2 Data Cleaning
Missing Values: The presence of missing values is checked and handled:
python

print("Missing values:\n", data.isnull().sum())
for i in data.columns:
    if data[i].isnull().sum() == 0:
        print(f'No of null values in {i}->{data[i].isnull().sum()}')
Duplicate Columns: Duplicates are detected and handled if necessary:
python
duplicates = find_duplicate_columns(data)
if duplicates:
    print("Duplicate columns found:")
    for original, duplicate_list in duplicates.items():
        print(f"{original} is duplicated with columns: {duplicate_list}")
else:
    print("No duplicate columns found.")

###2.3 Data Conversion
The target variable Heart Disease is converted from categorical to numerical values:

python

data['Heart Disease'] = data['Heart Disease'].map({'Presence': 1, 'Absence': 0})

### 2.4 Data Trimming
Outliers are trimmed using the 5th and 95th percentiles:
python

columns_to_trim = ['Chest pain type', 'Cholesterol', 'FBS over 120', 'Max HR', 'ST depression', 'Number of vessels fluro']
for column in columns_to_trim:
    lower_percentile = data[column].quantile(0.05)
    upper_percentile = data[column].quantile(0.95)
    data[column] = data[column].apply(lambda x: x if lower_percentile <= x <= upper_percentile else np.nan)
data = data.dropna(subset=columns_to_trim)

## 3. Data Transformation
### 3.1 Feature Scaling
Features are standardized using StandardScaler:
python

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(X_test)

## 4. Model Building and Evaluation
### 4.1 Models Used
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Support Vector Machine (SVM)

## 4.2 Model Training
### Models are trained using the training set:

python

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

logistic_model = LogisticRegression(random_state=42)
tree_model = DecisionTreeClassifier(random_state=42)
forest_model = RandomForestClassifier(random_state=42)
svm_model = SVC(probability=True, random_state=42)

logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

### 4.3 Model Prediction
Predictions are made on the test set:
python

logistic_pred = logistic_model.predict(X_test)
tree_pred = tree_model.predict(X_test)
forest_pred = forest_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

4.4 Model Evaluation
Models are evaluated using:

Classification Report
Accuracy Score
ROC Curve and AUC
Confusion Matrix
python

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

models = {"Logistic Regression": logistic_pred, "Decision Tree": tree_pred, "Random Forest": forest_pred, "SVM": svm_pred}

for model_name, y_pred in models.items():
    print(f"\nModel: {model_name}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='best')
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

## 5. Results
### 5.1 Model Performance
Logistic Regression: Accuracy = 0.84
Decision Tree: Accuracy = 0.72
Random Forest: Accuracy = 0.86
SVM: Accuracy = 0.77
5.2 Best Parameters (Random Forest)
The best parameters obtained for the Random Forest model are:

python

Best parameters found: {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}

5.3 Feature Importance
Feature importance from the Random Forest model is analyzed and visualized:

python

importances = forest_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(x.shape[1]), importances[indices], align='center')
plt.xticks(range(x.shape[1]), [x.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
```
6. Conclusion
The Random Forest model achieved the highest accuracy of 0.86, making it the most effective model for predicting heart disease in this project. The feature importance analysis highlights the most significant features contributing to the predictions.
