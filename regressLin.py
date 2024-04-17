import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Read the standardized dataset
df_standardized = pd.read_csv("out.csv", index_col=0)

# Define independent variables (features)
X = df_standardized[['Total Bags','Total Volume']]

# Define dependent variable (target)
y = df_standardized['AveragePrice']

# Scatterplot for AveragePrice vs Total Bags
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_standardized, x='Total Bags', y='AveragePrice', color='blue')
plt.title('Scatterplot: AveragePrice vs Total Bags')
plt.xlabel('Total Bags')
plt.ylabel('AveragePrice')
plt.show()

# Scatterplot for Total Volume vs Total Bags
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_standardized, x='Total Volume', y='AveragePrice', color='green')
plt.title('Scatterplot: Total Volume vs AveragePrice')
plt.xlabel('Total Volume')
plt.ylabel('AveragePrice')
plt.show()

# Calculate correlation matrix
correlation_matrix = X.corr()

# Print correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Calculate VIF for each independent variable
vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVIF:")
print(vif)

# Remove one of the variables with multicollinearity (Total Bags or Total Volume)
# Based on the provided results, let's remove Total Bags
X_processed = X.drop(columns=['Total Bags'])

# Fit the multiple linear regression model
X_processed = sm.add_constant(X_processed)  # Add a constant for the intercept term
model = sm.OLS(y, X_processed).fit()

# Print the model summary
print(model.summary())
