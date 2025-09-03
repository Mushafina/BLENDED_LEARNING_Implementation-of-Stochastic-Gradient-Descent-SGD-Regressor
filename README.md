# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Program to implement SGD Regressor for linear regression.
Developed by: R.Mushafina
RegisterNumber: 212224220067
# Importing necessary libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('C:/Users/admin/Downloads/CarPrice_Assignment (1) (1).csv')
print(data.head())
print(data.info())

# Data preprocessing 
# Dropping unnecessary columns and handling categorical variables 
data=data.drop(['CarName', 'car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)

# Splitting the data into features and target variable
x=data.drop('price',axis=1)
y=data['price']

# Standardizing the data 
scaler= StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

# Splitting the dataset into training and testing sets 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

# Fitting the model on the training data
sgd_model.fit(x_train, y_train)

# Making predictions
y_pred = sgd_model.predict(x_test)

# Evaluating model performance 
mse = mean_squared_error(y_test, y_pred)
r2= r2_score(y_test,y_pred)

# Print evaluation matrics 
print('Name: R.Mushafina')
print('Reg. No: 212224220067')
print("Mean Squared Error:",mse)
print("R-squared Score:",r2)

# Print model coefficients 
print("\nModel Coeeficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

# Visualizing actual vs predicted prices 
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test),max(y_test)], color='red')
plt.show()

```

## Output:
<img width="805" height="733" alt="image" src="https://github.com/user-attachments/assets/9aae25da-9a2c-4df3-b36d-78d0cb001806" />



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
