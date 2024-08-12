# Heart Disease Prediction

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
   - [Data Loading](#data-loading)
   - [Data Cleaning](#data-cleaning)
   - [Data Conversion](#data-conversion)
   - [Data Trimming](#data-trimming)
3. [Data Transformation](#data-transformation)
4. [Model Building and Evaluation](#model-building-and-evaluation)
   - [Models Used](#models-used)
   - [Model Training](#model-training)
   - [Model Prediction](#model-prediction)
   - [Model Evaluation](#model-evaluation)
5. [Results](#results)
   - [Model Performance](#model-performance)
   - [Best Parameters (Logistic Regression)](#best-parameters-logistic-regression)
6. [Model Saving](#model-saving)
7. [Conclusion](#conclusion)

---

## 1. Introduction
This document provides a comprehensive overview of the heart disease prediction project, covering data preparation, transformation, model building, and evaluation. The goal of the project is to predict heart disease using various machine learning models.

---

## 2. Data Preparation

### 2.1 Data Loading
The data is loaded from a CSV file:

```python
import pandas as pd
data = pd.read_csv('C:\\Users\\samee\\PycharmProjects\\Heart_Disease\\Heart_Disease_Prediction.csv')
```
2.2 Data Cleaning
Missing Values: The presence of missing values is checked and handled:
```python
print("Missing values:\n", data.isnull().sum())
for i in data.columns:
    if data[i].isnull().sum() == 0:
        print(f'No of null values in {i}->{data[i].isnull().sum()}')
```
Duplicate Columns: Duplicates are detected and handled if necessary:
```python
duplicates = find_duplicate_columns(data)
if duplicates:
    print("Duplicate columns found:")
    for original, duplicate_list in duplicates.items():
        print(f"{original} is duplicated with columns: {duplicate_list}")
else:
    print("No duplicate columns found.")
```
### 2.3 Data Conversion
The target variable Heart Disease is converted from categorical to numerical values:
```python
data['Heart Disease'] = data['Heart Disease'].map({'Presence': 1, 'Absence': 0})
```
### 2.4 Data Trimming
Outliers are trimmed using the 5th and 95th percentiles:

columns_to_trim = ['Chest pain type', 'Cholesterol', 'FBS over 120', 'Max HR', 'ST depression', 'Number of vessels fluro']
for column in columns_to_trim:
    lower_percentile = data[column].quantile(0.05)
    upper_percentile = data[column].quantile(0.95)
    data[column] = data[column].apply(lambda x: x if lower_percentile <= x <= upper_percentile else np.nan)
data = data.dropna(subset=columns_to_trim)
3. Data Transformation
### 3.1 Feature Scaling
Features are standardized using StandardScaler:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(X_test)
```

4. Model Building and Evaluation
### 4.1 Models Used
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Support Vector Machine (SVM)
4.2 Model Training
Models are trained using the training set:
```python
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
```
### 4.3 Model Prediction
Predictions are made on the test set:

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
```python
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
```
5. Results
### 5.1 Model Performance
Logistic Regression: Accuracy = 0.89
Decision Tree: Accuracy = 0.89
Random Forest: Accuracy = 0.87
SVM: Accuracy = 0.87
5.2 Best Parameters (Logistic Regression)
The best parameters found for the Logistic Regression model are:
```python
param_grid_logistic = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}
```
6. Model Saving
```python
import joblib
joblib.dump(logistic_model, 'logistic_regression_model.pkl')
```
7. Conclusion
The Logistic Regression model achieved the highest accuracy of 0.89, indicating its effectiveness in predicting heart disease in this project. The other models also performed well, with Random Forest and SVM showing comparable performance.
