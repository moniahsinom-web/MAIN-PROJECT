# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Step 1: Create sample dataset (normal + anomalies)
np.random.seed(42)

# Normal data
normal_data = np.random.normal(loc=50, scale=5, size=100)

# Anomalies
anomalies = np.random.uniform(low=80, high=100, size=10)

# Combine data
data = np.concatenate([normal_data, anomalies])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Transaction_Amount'])

# Step 2: Train Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)
df['Anomaly'] = model.fit_predict(df[['Transaction_Amount']])

# Step 3: Convert output (-1 = anomaly, 1 = normal)
df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})

print(df.head(15))

# Step 4: Plot results
plt.figure()

plt.scatter(df.index, df['Transaction_Amount'])
anomalies = df[df['Anomaly'] == 1]

plt.scatter(anomalies.index, anomalies['Transaction_Amount'])

plt.title("Anomaly Detection")
plt.xlabel("Index")
plt.ylabel("Transaction Amount")

plt.show()