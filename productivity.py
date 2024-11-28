import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
data = pd.read_csv(r'C:\Users\lenovo\Desktop\productivity+prediction+of+garment+employees\garments_worker_productivity.csv')

# Count total instances in the dataset
total_instances = data.shape[0]
print(f"Total number of instances in the dataset: {total_instances}")

# Display the first few rows of the dataset
print("Initial Data:")
print(data.head())

# 1. Data Cleaning
# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Visualize missing values before cleaning
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')
plt.title('Missing Values Count Before Cleaning')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Fill missing values in 'wip' and 'actual_productivity' with their means
data['wip'] = data['wip'].fillna(data['wip'].mean())
data['actual_productivity'] = data['actual_productivity'].fillna(data['actual_productivity'].mean())

# Check missing values after cleaning
missing_values_after = data.isnull().sum()
print("\nMissing Values After Cleaning:")
print(missing_values_after)

# 2. Exploratory Data Analysis (EDA)
# Summary statistics
summary_stats = data.describe()
print("\nSummary Statistics:")
print(summary_stats)

# Visualize distribution of actual productivity
plt.figure(figsize=(10, 6))
sns.histplot(data['actual_productivity'], bins=30, kde=True)
plt.title('Distribution of Actual Productivity')
plt.xlabel('Actual Productivity')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# 3. Data Aggregation and Grouping
# Average productivity by department
avg_productivity = data.groupby('department')['actual_productivity'].mean()
print("\nAverage Productivity by Department:")
print(avg_productivity)

# Total productivity per quarter
total_productivity_quarter = data.groupby('quarter')['actual_productivity'].sum()
print("\nTotal Productivity Over Quarters:")
print(total_productivity_quarter)

# 4. Visualization
# Bar chart for average productivity by department
plt.figure(figsize=(10, 6))
avg_productivity.plot(kind='bar', color='orange')
plt.title('Average Productivity by Department')
plt.ylabel('Average Productivity')
plt.xlabel('Department')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Line chart for total productivity over quarters
plt.figure(figsize=(10, 6))
total_productivity_quarter.plot(kind='line', marker='o')
plt.title('Total Productivity Over Quarters')
plt.ylabel('Total Productivity')
plt.xlabel('Quarter')
plt.grid()
plt.show()

# 5. Statistical Testing (Hypothesis Testing)
# Extract actual productivity for sewing and finishing departments
sewing_prod = data[data['department'].str.strip() == 'sweing']['actual_productivity']
finishing_prod = data[data['department'].str.strip() == 'finishing']['actual_productivity']

# Perform a t-test to compare actual productivity between sewing and finishing departments
t_stat, p_value = stats.ttest_ind(sewing_prod, finishing_prod, nan_policy='omit')

print("\nT-test Results:")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference in productivity between sewing and finishing departments.")
else:
    print("Fail to reject the null hypothesis: No significant difference in productivity between sewing and finishing departments.")

# Additional Results
mean_sewing = sewing_prod.mean()
mean_finishing = finishing_prod.mean()
print(f"\nMean Actual Productivity:")
print(f"Sewing Department: {mean_sewing:.4f}")
print(f"Finishing Department: {mean_finishing:.4f}")

# 6. Correlation Analysis
# Select only numeric columns for correlation analysis
numeric_data = data.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_data.corr()  # Calculate correlation matrix

# Plotting the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# 7. Key Insights
insights = {
    "Highest Average Productivity": avg_productivity.idxmax(),
    "Lowest Average Productivity": avg_productivity.idxmin(),
    "Peak Productivity Quarter": total_productivity_quarter.idxmax(),
    "Lowest Productivity Quarter": total_productivity_quarter.idxmin(),
}

print("\nKey Insights:")
for key, value in insights.items():
    print(f"{key}: {value}")