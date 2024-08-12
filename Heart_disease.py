import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
data=pd.read_csv('C:\\Users\\lizyd\\PycharmProjects\\Heart_disease_prediction\\Heart_Disease_Prediction.csv')
# print(data.head())
# Data cleaning
# check the null values
# print("Missing values:\n", data.isnull().sum())
# for i in data.columns:
#     if data[i].isnull().sum()==0:
#         print(f'No of null values in {i}->{data[i].isnull().sum()}')
c=[]
# d=[]
# for i in data.columns:
#     if data[i].dtype=='object':
#         c.append(i)
#
#     else:
#         d.append(i)
# print(c)
# print(d)
# def find_duplicate_columns(data):
#     # Dictionary to store column hashes
#     col_hashes = {}
#     duplicates = {}
#
#     for col in data.columns:
#         # Convert column to a frozenset of (value, index) tuples to handle column order
#         col_hash = frozenset(data[col].items())
#
#         if col_hash in col_hashes:
#             if col_hashes[col_hash] in duplicates:
#                 duplicates[col_hashes[col_hash]].append(col)
#             else:
#                 duplicates[col_hashes[col_hash]] = [col]
#         else:
#             col_hashes[col_hash] = col
#
#     return duplicates
#
#
# # Check for duplicates
# duplicates = find_duplicate_columns(data)
#
# if duplicates:
#     print("Duplicate columns found:")
#     for original, duplicate_list in duplicates.items():
#         print(f"{original} is duplicated with columns: {duplicate_list}")
# else:
#     print("No duplicate columns found.")

# List to store names of duplicate columns
# duplicate_columns = []

# Compare each column with every other column
# for i in range(len(data.columns)):
#     for j in range(i + 1, len(data.columns)):
#         if data.iloc[:, i].equals(data.iloc[:, j]):
#             duplicate_columns.append(data.columns[j])
#
# # Remove duplicates from the list (in case of multiple duplicates)
# duplicate_columns = list(set(duplicate_columns))
#
# # Print the result
# if duplicate_columns:
#     print('Duplicate Columns:', duplicate_columns)
# else:
#     print('No duplicate columns found.')
# Convert categorical to numerical data
# unique_values=data['Heart Disease'].unique()
# print(unique_values)
# As this is the nominal data we can use both the mapping(dependent) and the onehot encoder to this
# one_hot=OneHotEncoder()
# one_hot.fit(data['Heart Disease'])
data['Heart Disease']=data['Heart Disease'].map({'Presence': 1, 'Absence': 0})
# print(data.head())

# print(x.head())
# print(y.head())

# print(f'No of rows for training purpose:{len(x_train)}')
# print(f'No of rows for testing purpose:{len(x_test)}')
# print(f'No of rows for training purpose:{len(y_train)}')
# print(f'No of rows for testing purpose:{len(y_test)}')
# Step 3: Feature Selection (Correlation Matrix)
# Create a figure with multiple subplots, one for each column
# plt.figure(figsize=(12, 8))
# for i, column in enumerate(data.columns, 1):
#     plt.subplot(len(data.columns) // 3 + 1, 3, i)  # Adjust the grid size based on the number of columns
#     sns.boxplot(y=data[column])
#     plt.xlabel('Value')
#     plt.ylabel(column)  # Use column name as the ylabel
#     plt.title(column)
#     plt.tight_layout()
# #
# # plt.show()
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# plt.figure(figsize=(12, 8))
# for i, column in enumerate(data.columns, 1):
#     plt.subplot(len(data.columns) // 3 + 1, 3, i)  # Adjust the grid size based on the number of columns
#     sns.kdeplot(data[column], fill=True)
#     plt.xlabel(column)  # Use column name as the xlabel
#     plt.ylabel('Density')  # Add ylabel for density
#     plt.title(column)  # Add title for each subplot
#     plt.tight_layout()
#
# plt.show()
# print(data['Chest pain type'].value_counts())
# # Ensure no zero or negative values for log transformation
# data['Chest pain type'] = data['Chest pain type'] + 1  # Adding 1 to avoid log(0)
# data['Chest Pain Type Log'] = np.log(data['Chest pain type'])
# print(data[['Chest pain type', 'Chest Pain Type Log']].head())
# plt.figure(figsize=(10, 6))
# sns.boxplot(x=data['Chest Pain Type Log'])
# plt.title('Boxplot of Log-Transformed Chest Pain Type')
# plt.xlabel('Log-Transformed Chest Pain Type')
# plt.show()
# I/N Transformation
# data['Chest Pain Type I/N'] = (data['Chest pain type'] - data['Chest pain type'].min()) / (data['Chest pain type'].max() - data['Chest pain type'].min())
# Log Transformation
# Ensure no zero or negative values for log transformation
# data['BP'] = data['BP'] + 1  # Adding 1 to avoid log(0)
# data['BP Log'] = np.log(data['BP'])

# I/N Transformation
# data['BP I/N'] = (data['BP'] - data['BP'].min()) / (data['BP'].max() - data['BP'].min())
#
# # Trimming Based on Percentiles
# lower_percentile = data['BP'].quantile(0.05)
# upper_percentile = data['BP'].quantile(0.95)

# # Remove outliers by trimming
# data_trimmed = data[(data['BP'] >= lower_percentile) & (data['BP'] <= upper_percentile)]
#
# # Check the updated distribution
# print("\nDistribution after trimming:")
# print(data_trimmed['BP'].value_counts())
#
# # Display transformed data
# print("\nTransformed data sample:")
# print(data[['BP', 'BP Log', 'BP I/N']].head())
# List of columns to apply the transformation
columns_to_trim = ['Chest pain type', 'Cholesterol', 'BP','FBS over 120', 'Max HR', 'ST depression', 'Number of vessels fluro']

# Apply the 5th and 95th percentile trimming to each column
for column in columns_to_trim:
    lower_percentile = data[column].quantile(0.05)
    upper_percentile = data[column].quantile(0.95)

    # Trim the column
    # data[f'{column} Trimmed'] = data[column].apply(lambda x: x if lower_percentile <= x <= upper_percentile else np.nan)
    data[column] = data[column].apply(lambda x: x if lower_percentile <= x <= upper_percentile else np.nan)

# Drop NaN values (rows that had outliers removed) if necessary
# data_trimmed = data.dropna(subset=[f'{col} Trimmed' for col in columns_to_trim])
data = data.dropna(subset=columns_to_trim)
# Display the trimmed data sample
# print("\nTrimmed data sample:")
# print(data_trimmed[[f'{col} Trimmed' for col in columns_to_trim]].head())

print("\nUpdated data sample after replacing original columns with trimmed values:")
print(data[columns_to_trim].head())
print(data.head())
# Create boxplots for each trimmed column to check whether the outliers are cleared
# plt.figure(figsize=(18, 12))
#
# # Iterate through each column to generate the box plots
# for i, column in enumerate(columns_to_trim, 1):
#     plt.subplot(2, 3, i)
#     # sns.boxplot(data_trimmed[f'{column} Trimmed'])
#     sns.boxplot(data[column])
#     plt.title(f'Boxplot of Trimmed {column}')
#     plt.xlabel(f'{column} (Trimmed)')
#
# plt.tight_layout()
# plt.show()

# # Original Values
# plt.subplot(1, 4, 1)
# sns.boxplot(data['BP'])
# plt.title('Original Chest Pain Type')
# plt.xlabel('Chest Pain Type')

# # Log-Transformed Values
# plt.subplot(1, 4, 2)
# sns.boxplot(data['BP Log'])
# plt.title('Log-Transformed Chest Pain Type')
# plt.xlabel('Log-Transformed Chest Pain Type')

# # I/N Transformed Values
# plt.subplot(1, 4, 3)
# sns.boxplot(data['BP I/N'])
# plt.title('I/N Transformed Chest Pain Type')
# plt.xlabel('I/N Transformed Chest Pain Type')

# plt.subplot(1, 4, 4)
# sns.boxplot(data_trimmed['BP'])
# plt.title('5th and 95th quantile BP')
# plt.xlabel('5th and 95th quantile BP')
# plt.tight_layout()
# plt.show()
x=data.iloc[ : , :-1]#independent
y=data.iloc[ : , -1]#dependent
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
# Step 4.1: Logistic Regression
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)
# Step 4.2: Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

# Step 4.3: Random Forest
forest_model = RandomForestClassifier(random_state=42)
forest_model.fit(X_train, y_train)
forest_pred = forest_model.predict(X_test)

# Step 4.4: Support Vector Machine (SVM)
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Step 5: Model Evaluation
models = {
    "Logistic Regression": logistic_pred,
    "Decision Tree": tree_pred,
    "Random Forest": forest_pred,
    "SVM": svm_pred
}
# Evaluate each model
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

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
# Define the parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }
#
# # Initialize the GridSearchCV object
# grid_search = GridSearchCV(estimator=forest_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
#
# # Fit the model
# grid_search.fit(X_train, y_train)
#
# # Get the best parameters
# print("Best parameters found: ", grid_search.best_params_)
#
# # Update the model with the best parameters
# forest_model = grid_search.best_estimator_
# # Feature importance from Random Forest
# importances = forest_model.feature_importances_
# indices = np.argsort(importances)[::-1]
#
# # Print the feature ranking
# print("Feature ranking:")
#
# for i in range(x.shape[1]):
#     print(f"{i + 1}. Feature {indices[i]} ({importances[indices[i]]:.4f})")
#
# # Plot the feature importances
# plt.figure(figsize=(12, 6))
# plt.title("Feature Importance")
# plt.bar(range(x.shape[1]), importances[indices], align='center')
# plt.xticks(range(x.shape[1]), [x.columns[i] for i in indices], rotation=90)
# plt.tight_layout()
# plt.show()
# # Save the model
# joblib.dump(forest_model, 'random_forest_model.pkl')
#
# # Load the model later for prediction
# # forest_model = joblib.load('random_forest_model.pkl')
# If Logistic Regression requires tuning (for example, the regularization parameter):
# Define the parameter grid for Logistic Regression
param_grid_logistic = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

# Initialize GridSearchCV for Logistic Regression
grid_search_logistic = GridSearchCV(estimator=logistic_model, param_grid=param_grid_logistic, cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search_logistic.fit(X_train, y_train)

# Get the best parameters
print("Best parameters for Logistic Regression found: ", grid_search_logistic.best_params_)

# Update the model with the best parameters
best_logistic_model = grid_search_logistic.best_estimator_

# Evaluate the tuned Logistic Regression model
best_logistic_pred = best_logistic_model.predict(X_test)
print("\nTuned Logistic Regression Model:")
print("Classification Report:\n", classification_report(y_test, best_logistic_pred))
print("Accuracy:", accuracy_score(y_test, best_logistic_pred))

# Save the tuned Logistic Regression model
joblib.dump(best_logistic_model, 'logistic_regression_model.pkl')