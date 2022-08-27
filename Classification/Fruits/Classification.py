# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:08:55 2019

@author: abhinav
"""
import pandas as pd
import matplotlib.pyplot as plt

fruits = pd.read_csv('fruit_data.txt',sep='\t')
print (fruits.head())
print (fruits.groupby('fruit_name').size())
print (fruits.shape)

import seaborn as sns
sns.countplot(fruits['fruit_name'],label='Count')
plt.show()

import pylab as pl
fruits.drop('fruit_label',axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle('Histogram for each numeric input variable')
plt.savefig('fruits_hist')
plt.show()

feature_names = ['mass','width','height','color_score']
X=fruits[feature_names]     #Predictor Variable
y=fruits['fruit_label']     #Target Variable

#Data Splicing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

#Data Normalizing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print ('Accuracy of K-NN Classifier on training dataset is: {:.2f}'.format(knn.score(X_train,y_train)))
print ('Accuracy of K-NN Classifier on testing dataset is: {:.2f}'.format(knn.score(X_test,y_test)))

#Confusion matrix for knn classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

df = pd.DataFrame({'Actual':y_test,'Predicted':pred.flatten()})
df.to_excel('ClassificationResult.xlsx',sheet_name='Actual and Predicted')
print (df)
