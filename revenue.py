# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("revenue_data.csv")

# Display data
print(df.head())

# Convert categorical column (Region) to numbers
le = LabelEncoder()
df['Region'] = le.fit_transform(df['Region'])

# Define features and target
X = df[['Marketing_Spend', 'Customers', 'Region']]
y = df['Revenue']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict new data
new_data = [[1500, 60, 1]]  # Example input
prediction = model.predict(new_data)

print("Predicted Revenue:", prediction[0])