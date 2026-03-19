# =====================================
# Medical Diagnosis Prediction System
# =====================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("medical_data.csv")

print(data.head())

# -------------------------------
# Features and Target
# -------------------------------
X = data.drop("Outcome", axis=1)   # Input features
y = data["Outcome"]                # Target (0 or 1)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# Prediction Function
# -------------------------------
def predict_disease(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "⚠️ Disease Detected"
    else:
        return "✅ No Disease Detected"

# -------------------------------
# Sample Patient Data
# -------------------------------
# Example values:
# [Pregnancies, Glucose, BloodPressure, BMI, Age]

sample_patient = [2, 150, 85, 33.6, 45]

print("\nPrediction Result:")
print(predict_disease(sample_patient))
