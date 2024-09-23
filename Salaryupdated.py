# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:54:55 2024

@author: roy62
"""

import numpy as np 	
import matplotlib.pyplot as plt
import pandas as pd	
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
# Load the dataset
salaryupdated = pd.read_csv(r"E:\Data Science & AI\Dataset files\Salary_Data.csv")  # Load the dataset
salaryupdated

print("Salaryupdated Shape:", salaryupdated.shape)# Check the shape of the dataset

# Split the data into independent and dependent variables
x= salaryupdated.iloc[:, :-1].values
y= salaryupdated.iloc[:, 1].values 

# Split the dataset into training and testing sets (80-20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

regressor = LinearRegression()  # Train the model
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)  # Predict the test set

#comparision for y_test vs y_pred
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

# Visualize the training set
plt.scatter(x_train, y_train, color='green') 
plt.plot(x_train, regressor.predict(x_train), color='black')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualize the test set
plt.scatter(x_test, y_test, color='red') 
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict salary for 12 and 20 years of experience using the trained model
y_12 = regressor.predict([[12]])
print(f"Predicted salary for 12 years of experience: ${y_12[0]:,.2f}")

y_20 = regressor.predict([[20]])
print(f"Predicted salary for 20 years of experience: ${y_20[0]:,.2f}")

# Check model performance
bias = regressor.score(x_train, y_train)
print(f"Training Score (R^2): {bias:.2f}")

variance = regressor.score(x_test, y_test)
print(f"Testing Score (R^2): {variance:.2f}")

train_mse = mean_squared_error(y_train, regressor.predict(x_train))
print(f"Training MSE: {train_mse:.2f}")

test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk
filename = 'Salaryupdated_linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as Salaryupdated_linear_regression_model.pkl")

import os 
print(os.getcwd())

from sklearn.metrics import r2_score,mean_squared_error

R2=r2_score(y_test,y_pred)
print("R-sqaure:",R2)

MSE=mean_squared_error(y_test,y_pred)
#MSE**(1/2)
print("MSE:",MSE)

RMSE=np.sqrt(MSE)
#accuracy_score(y_test,y_predictions) # it is a regression tech
print("RMSE:",RMSE)

#regression table code
# introduce to OLS & stats.api
from statsmodels.api import OLS
OLS(y_train,x_train).fit().summary()



