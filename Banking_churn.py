import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC as svc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

import numpy as np

data=pd.read_csv('https://raw.githubusercontent.com/regan-mu/ADS-April-2022/main/Assignments/Assignment%202/banking_churn.csv')
st.write(data.head())

st.write(data.info())

st.write(data.isna().sum())

#Removing columns that have no effect on customer churn
st.write(data.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True))
st.write(data.head())

st.write(data.describe())

st.write(data.groupby('Exited').mean())

#separating the features and labels
x= data.drop(columns='Exited',axis=1)
y=data['Exited']

st.write(print(x))
st.write(print(y))

#splitting the data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

# convert categorical columns in text to numerical values i.e Gender and Geography
tempdata=data.drop(['Geography','Gender'], axis=1)
Geography=pd.get_dummies(data.Geography).iloc[:,1:]
Gender=pd.get_dummies(data.Gender).iloc[:,1:]

data=pd.concat([tempdata,Geography,Gender], axis=1)
st.write(data.head())

#separating the features and labels
x= data.drop(columns='Exited',axis=1)
y=data['Exited']

#splitting the data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

print(x)
print(y)

st.write(x.shape, x_test.shape, x_train.shape)

x= data.drop(columns='Exited',axis=1)
y=data['Exited']
#Training and modelling the train data

classifier=rfc(n_estimators=200, random_state=0)
classifier.fit(x_train,y_train)

#accuracy score
#Accuracy score on the training data

x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train,y_train)
