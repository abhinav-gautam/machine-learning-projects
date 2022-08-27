# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:35:13 2019

@author: abhinav
"""
#Linear Regression Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('weather.csv')
print (dataset.shape)
print (dataset.describe())

dataset.plot(x='MinTemp', y='MaxTemp',style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

#Average of MaxTemp
plt.figure(figsize=(10,15))
plt.tight_layout()
seabornInstance.distplot(dataset['MaxTemp'])
plt.show()

#Data Splicing
X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Training the Model
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Intercept and Coefficient
print ('Intercept',regressor.intercept_)
print ('Coefficient',regressor.coef_)

#Predicting for X_test
y_pred = regressor.predict(X_test)

#Comparison Between Actual and Predicted
df = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
#df.to_excel('LinearRegressionResult.xlsx',sheet_name='Actual and Predicted')
print (df)

df1=df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major',linestyle='-',linewidth='0.5', color='green')
plt.grid(which='minor',linestyle=':',linewidth='0.5', color='black')
plt.show()

plt.scatter(X_test,y_test,color='grey')
plt.plot(X_test,y_pred,linewidth=2)
plt.show()

#Performance
print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error:",metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))