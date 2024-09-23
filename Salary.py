# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:19:29 2024

@author: roy62
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
salary = pd.read_csv(r"E:\Data Science & AI\Dataset files\Salary_Data.csv")# Load the dataset
salary


print("Salary Shape:", salary.shape)# Check the shape of the dataset

# Feature selection (independent variable X and dependent variable y)
x = salary.iloc[:,:-1]  # Years of experience (Independent variable)
y = salary.iloc[:,-1]   # Salary (Dependent variable)

# Split the dataset into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=20,random_state=0)

# You don't need to reshape y_train, as it's the target variable
# Fit the Linear Regression model to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)# Predicting the results for the test set

# Compare predicted and actual salaries from the test set
comparison = pd.DataFrame({'Actual':y_test,'predicated':y_pred})
print(comparison)

# Visualizing the Training set results
plt.scatter(x_test,y_test,color ='red')# Real salary data (training)
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(x_test, y_test, color = 'red')  # Real salary data (testing)
plt.plot(x_train, regressor.predict(x_train), color = 'blue')  # Regression line from training set
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Optional: Output the coefficients of the linear model
m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

future_pred = m_slope * 12 + c_intercept
print(future_pred)

future_pred = m_slope * 20 + c_intercept
print(future_pred)


