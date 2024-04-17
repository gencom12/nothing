import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest

# Read the dataset
df = pd.read_csv('avocado.csv')

# Display dataset information
print("Dataset Information:")
print(df.info())

# Display the first 5 rows of the dataset
print("\nDataset:")
print(df.head(5))

df.drop('Unnamed: 0',axis=1,inplace=True)
df['Date']=pd.to_datetime(df['Date'])
df['Month']=df['Date'].apply(lambda x:x.month)
df['Day']=df['Date'].apply(lambda x:x.day)

# List categorical features
categorical_features = df.select_dtypes(include='object').columns
print("\nCategorical Features:")
print(categorical_features.tolist())

# List numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
print("\nNumerical Features:")
print(numerical_features.tolist())

# Percentage of missing values in categorical features
missing_categorical = df[categorical_features].isnull().sum() * 100 / len(df)
print("\nPercentage of missing values in categorical features:")
print(missing_categorical)

# Percentage of missing values in numerical features
missing_numerical = df[numerical_features].isnull().sum() * 100 / len(df)
print("\nPercentage of missing values in numerical features:")
print(missing_numerical)

# Outlier detection using Isolation Forest
clf = IsolationForest(random_state=42, contamination='auto')
outliers_removed_df = df.copy()
outliers_removed_df['outlier'] = clf.fit_predict(outliers_removed_df[numerical_features])
outliers_removed_df = outliers_removed_df[outliers_removed_df['outlier'] == 1]
outliers_removed_df.drop(columns='outlier', inplace=True)

# Distribution plots for numerical features after handling outliers
plt.figure(figsize=(15, 7))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(1, len(numerical_features), i)
    sns.histplot(outliers_removed_df[feature], kde=True, color='salmon')
    plt.title(f'Distribution of {feature}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# Standardizing numerical features
scaler = MinMaxScaler()
outliers_removed_df[numerical_features] = scaler.fit_transform(outliers_removed_df[numerical_features])

# Save the processed dataset to a CSV file
outliers_removed_df.to_csv("out.csv", index=False)

# EDA - No need to change this section
outliers_removed_df['type'] = outliers_removed_df['type'].astype('category')

plt.figure(figsize=(10, 6))
sns.lineplot(data=outliers_removed_df, x='Date', y='AveragePrice', color='darkorange')
plt.xlabel('Date')
plt.ylabel('Average Price ($)')
plt.title('Average Price of Avocados over Time')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=outliers_removed_df, x='Date', y='Total Volume', color='green')
plt.xlabel('Date')
plt.ylabel('Total Volume')
plt.title('Total Volume of Avocados over Time')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=outliers_removed_df, x='Date', y='Total Bags', color='blue')
plt.xlabel('Date')
plt.ylabel('Total Bags')
plt.title('Total Bags of Avocados over Time')
plt.show()

bags_columns = ['Small Bags', 'Large Bags', 'XLarge Bags']
plt.figure(figsize=(10, 6))
outliers_removed_df.set_index('Date')[bags_columns].plot(kind='bar', stacked=True, color=['skyblue', 'salmon', 'green'])
plt.xlabel('Date')
plt.ylabel('Number of Bags')
plt.title('Types of Bags over Time')
plt.savefig('Types_of_Bags_vs_Date.png')
plt.close()

plt.figure(figsize=(8, 8))
avocado_counts = outliers_removed_df['type'].value_counts()
plt.pie(avocado_counts, labels=avocado_counts.index, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Distribution of Avocados by Type')
plt.savefig('Avocados_by_Type.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='AveragePrice', y='region', data=outliers_removed_df, hue='region', legend=False, palette='pastel')
plt.xlabel('Average Price ($)')
plt.ylabel('Region')
plt.title('Distribution of Average Avocado Prices by Region')
plt.show()
