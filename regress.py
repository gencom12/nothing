import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm


'''I decided to use three regression models just in case and used the one which provided the best results 
based on the RMSE(Root Mean Square Error) value. The Random Forest Regression turned out to give the best
results for the dataset'''

# Read the data prepped csv file into a dataframe
df = pd.read_csv("out.csv")

# Convert relevant columns to numeric dtype
df['AveragePrice'] = pd.to_numeric(df['AveragePrice'], errors='coerce')

# Group by 'Date' and calculate the mean of numeric columns
df['AveragePrice'] = pd.to_numeric(df['AveragePrice'], errors='coerce')

# Convert 'Date' column to datetime dtype
df['Date'] = pd.to_datetime(df['Date'])

# Group by 'Date' and calculate the mean of 'AveragePrice'
byDate = df.groupby('Date')['AveragePrice'].mean().reset_index()

# Plotting the average price over time
plt.figure(figsize=(12, 8))
plt.plot(byDate['Date'], byDate['AveragePrice'])
plt.title('Average Price Over Time')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.show()

# How dataset features are correlated with each other?
df['AveragePrice'] = pd.to_numeric(df['AveragePrice'], errors='coerce')

# Drop non-numeric columns
df_numeric = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
corr_matrix = df_numeric.corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
plt.title('Correlation Matrix')
plt.show()

df['region'].nunique()

df['type'].nunique()

df_final = pd.get_dummies(df.drop(['region', 'Date'], axis=1), drop_first=True)
df_final.head()
df_final.tail()

X = df_final.iloc[:, 1:14]
y = df_final['AveragePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print('Linear Regression:')
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
plt.figure()
plt.scatter(x=y_test, y=pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# Decision Tree Regression
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
pred = dtr.predict(X_test)
plt.figure()
plt.scatter(x=y_test, y=pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
print('Decision Tree Regression:')
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
plt.show()

# Random Forest Regression
rdr = RandomForestRegressor()
rdr.fit(X_train, y_train)
pred = rdr.predict(X_test)
print('Random Forest Regression:')
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
sns.distplot((y_test - pred), bins=50)
plt.show()

# Predicted vs Actual Values
data = pd.DataFrame({'Y Test': y_test, 'Pred': pred}, columns=['Y Test', 'Pred'])
print(data)
plt.figure()
sns.lmplot(x='Y Test', y='Pred', data=data, palette='rainbow')
data.head()
plt.show()