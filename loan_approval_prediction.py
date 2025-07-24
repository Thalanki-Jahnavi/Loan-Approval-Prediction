# loan_approval_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("loan_approval_dataset.csv")

# Fill missing values
data['no_of_dependents'].fillna(data['no_of_dependents'].mode()[0], inplace=True)
data['education'].fillna(data['education'].mode()[0], inplace=True)
data['self_employed'].fillna(data['self_employed'].mode()[0], inplace=True)
data['loan_amount'].fillna(data['loan_amount'].median(), inplace=True)
data['loan_term'].fillna(data['loan_term'].median(), inplace=True)
data['cibil_score'].fillna(data['cibil_score'].median(), inplace=True)
data['loan_status'].fillna(data['loan_status'].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
data['education'] = le.fit_transform(data['education'])
data['self_employed'] = le.fit_transform(data['self_employed'])
data['loan_status'] = le.fit_transform(data['loan_status'])

# Drop ID column
data.drop('loan_id', axis=1, inplace=True)

# Split features and target
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

