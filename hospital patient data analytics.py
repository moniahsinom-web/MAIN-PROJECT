import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample Hospital Patient Data
data = {
    "Patient_ID": [1, 2, 3, 4, 5, 6, 7, 8],
    "Age": [25, 40, 35, 60, 50, 45, 30, 70],
    "Gender": ["F", "M", "F", "M", "F", "M", "F", "M"],
    "Department": ["Cardiology", "Orthopedics", "Neurology",
                   "Cardiology", "Orthopedics", "Neurology",
                   "Cardiology", "Orthopedics"]
}

df = pd.DataFrame(data)

# Seaborn style
sns.set(style="whitegrid")

# Plot 1: Patient count by department
plt.figure(figsize=(6,4))
sns.countplot(x="Department", data=df)
plt.title("Patient Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Number of Patients")
plt.show()

# Plot 2: Age distribution of patients
plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=6, kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Patient Count")
plt.show()
