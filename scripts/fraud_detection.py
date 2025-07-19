import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
fraud_data = pd.read_csv('Fraud_Data.csv')
ip_country = pd.read_csv('IpAddress_to_Country.csv')
creditcard = pd.read_csv('creditcard.csv')

# Task 1: Data Analysis and Preprocessing

# Handle Missing Values
fraud_data = fraud_data.dropna()  # Drop rows with missing values
creditcard = creditcard.dropna()

# Data Cleaning
fraud_data = fraud_data.drop_duplicates()
creditcard = creditcard.drop_duplicates()

# Correct Data Types
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)

# Merge Datasets for Geolocation Analysis
def map_ip_to_country(ip, ip_country_df):
    for _, row in ip_country_df.iterrows():
        if row['lower_bound_ip_address'] <= ip <= row['upper_bound_ip_address']:
            return row['country']
    return 'Unknown'

fraud_data['country'] = fraud_data['ip_address'].apply(lambda x: map_ip_to_country(x, ip_country))

# Feature Engineering for Fraud_Data
fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600

# Transaction Frequency and Velocity
fraud_data['transaction_count'] = fraud_data.groupby('user_id')['purchase_time'].transform('count')
fraud_data['velocity'] = fraud_data['purchase_value'] / (fraud_data['time_since_signup'] + 1)  # Avoid division by zero

# EDA: Class Distribution
print("Fraud_Data Class Distribution:")
print(fraud_data['class'].value_counts(normalize=True))
print("\nCreditcard Class Distribution:")
print(creditcard['Class'].value_counts(normalize=True))

# Handle Class Imbalance with SMOTE (applied later during model training)

# Encode Categorical Features and Scale Numerical Features
categorical_features = ['source', 'browser', 'sex', 'country']
numerical_features = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 'time_since_signup', 'transaction_count', 'velocity']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Task 2: Model Building and Training

# Prepare Fraud_Data
X_fraud = fraud_data.drop(['class', 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address'], axis=1)
y_fraud = fraud_data['class']
X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)

# Prepare Creditcard Data
X_credit = creditcard.drop('Class', axis=1)
y_credit = creditcard['Class']
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_fraud_smote, y_train_fraud_smote = smote.fit_resample(preprocessor.fit_transform(X_train_fraud), y_train_fraud)
X_train_credit_smote, y_train_credit_smote = smote.fit_resample(X_train_credit, y_train_credit)

# Define Models
logreg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train and Evaluate Models for Fraud_Data
pipeline_logreg_fraud = Pipeline([('preprocessor', preprocessor), ('classifier', logreg)])
pipeline_rf_fraud = Pipeline([('preprocessor', preprocessor), ('classifier', rf)])

pipeline_logreg_fraud.fit(X_train_fraud, y_train_fraud)
pipeline_rf_fraud.fit(X_train_fraud, y_train_fraud)

# Evaluate Models
def evaluate_model(model, X_test, y_test, dataset_name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{dataset_name} Evaluation:")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    return auc_pr, f1

# Fraud_Data Evaluation
auc_pr_logreg_fraud, f1_logreg_fraud = evaluate_model(pipeline_logreg_fraud, X_test_fraud, y_test_fraud, "Fraud_Data Logistic Regression")
auc_pr_rf_fraud, f1_rf_fraud = evaluate_model(pipeline_rf_fraud, X_test_fraud, y_test_fraud, "Fraud_Data Random Forest")

# Train and Evaluate Models for Creditcard Data
logreg.fit(X_train_credit_smote, y_train_credit_smote)
rf.fit(X_train_credit_smote, y_train_credit_smote)

# Creditcard Evaluation
auc_pr_logreg_credit, f1_logreg_credit = evaluate_model(logreg, X_test_credit, y_test_credit, "Creditcard Logistic Regression")
auc_pr_rf_credit, f1_rf_credit = evaluate_model(rf, X_test_credit, y_test_credit, "Creditcard Random Forest")

# Model Selection Justification
best_model = 'Random Forest' if (auc_pr_rf_fraud + auc_pr_rf_credit) > (auc_pr_logreg_fraud + auc_pr_logreg_credit) else 'Logistic Regression'
print(f"\nBest Model: {best_model}")
print("Justification: Random Forest typically performs better on imbalanced datasets due to its ability to capture complex patterns and interactions between features, which is critical for fraud detection.")

# Task 3: Model Explainability with SHAP
# Use Random Forest for explainability (assuming it performs better)
X_test_fraud_transformed = preprocessor.transform(X_test_fraud)
explainer = shap.TreeExplainer(pipeline_rf_fraud.named_steps['classifier'])
shap_values = explainer.shap_values(X_test_fraud_transformed)

# Generate SHAP Summary Plot
feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
shap.summary_plot(shap_values[1], X_test_fraud_transformed, feature_names=feature_names)
plt.savefig('shap_summary_plot.png')

# Interpretation
print("\nSHAP Summary Plot Interpretation:")
print("The SHAP summary plot shows the impact of each feature on the model's prediction of fraud. Features like 'time_since_signup' and 'velocity' are likely key drivers, indicating that rapid transactions after signup are strong indicators of fraud. Categorical features like 'country' and 'source' also contribute, suggesting geographic and acquisition channel patterns in fraudulent behavior.")