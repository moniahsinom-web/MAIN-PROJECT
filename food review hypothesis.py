import pandas as pd
from scipy.stats import pearsonr, ttest_ind, chi2_contingency

# Sample dataset
df = pd.DataFrame({
    'Rating': [5, 2, 4, 1, 3, 5, 2, 4],
    'Delivery_Time': [25, 60, 30, 75, 45, 20, 65, 35],
    'Discount': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
})

# 1. Delivery Time vs Rating (Correlation)
corr, p1 = pearsonr(df['Delivery_Time'], df['Rating'])

# 2. Discount vs Rating (T-Test)
rating_yes = df[df['Discount'] == 'Yes']['Rating']
rating_no = df[df['Discount'] == 'No']['Rating']
t_stat, p2 = ttest_ind(rating_yes, rating_no)

# 3. Review Sentiment vs Rating (Chi-Square)
contingency = [[20, 5], [8, 17]]
chi2, p3, dof, expected = chi2_contingency(contingency)

# Results
print("Delivery Time vs Rating -> Correlation:", corr, "P-value:", p1)
print("Discount vs Rating -> P-value:", p2)
print("Sentiment vs Rating -> P-value:", p3)
