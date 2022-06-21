import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
data=pd.read_csv('https://raw.githubusercontent.com/regan-mu/ADS-April-2022/main/Assignments/Assignment%202/banking_churn.csv')
data

st.write(data.info())
st. write(data.describe())

st.write(data.isna().sum())

#Removing columns that have no effect on customer churn
st.write(data.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True))

st.write(data.head())

# To find out if Gender has any effect on the output

counts= data.groupby(['Gender','Exited']).Exited.count().unstack()
st.write(counts.plot(kind='bar',stacked=True))
st.pyplot()
st.write(counts)

#to find out the effect of Geography on the output

counts= data.groupby(['Geography','Exited']).Exited.count().unstack()
st.write(counts.plot(kind='bar',stacked=True))
st.pyplot()
st.write(counts)

#to find out the effect of NumOfProducts on the output
counts= data.groupby(['NumOfProducts','Exited']).Exited.count().unstack()
counts.plot(kind='bar',stacked=True)
st.pyplot()
st.write(counts)

# convert categorical columns in text to numerical values i.e Gender and Geography
tempdata=data.drop(['Geography','Gender'], axis=1)
st.write(tempdata.head(2))

st.write("Create one hot encoded columns for Geography and Gender")
Geography=pd.get_dummies(data.Geography).iloc[:,1:]
Gender=pd.get_dummies(data.Gender).iloc[:,1:]
data=pd.concat([tempdata,Geography,Gender], axis=1)
st.write(data.head())

dataset_features=data.drop(['Exited'],axis=1)
dataset_labels=data['Exited']

st.write(dataset_features.head())

from sklearn.model_selection import train_test_split, GridSearchCV
train_features, test_features, train_labels, test_labels = train_test_split(dataset_features, dataset_labels, test_size=0.2, random_state=21)
st.write("shape of train and test splits")
train_features.shape, test_features.shape, train_labels.shape, test_labels.shape

st.subheader("Training and evaluation of ML models")
st.write("Training the Model using Random Forest Algorithm")

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC as svc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

#RandomForestClassifier
rfc_object=rfc(n_estimators=200, random_state=0)
rfc_object.fit(train_features,train_labels)
predicted_labels=rfc_object.predict(test_features)

st.write(classification_report(test_labels, predicted_labels))
st.write(confusion_matrix(test_labels, predicted_labels))
st.write(accuracy_score(test_labels, predicted_labels))

#Support Vector Machines (SVM)

svc_object = svc(kernel='rbf', degree=8)
svc_object.fit(train_features, train_labels)
predicted_labels = svc_object.predict(test_features)

st.write(classification_report(test_labels, predicted_labels))
st.write(confusion_matrix(test_labels, predicted_labels))
st.write(accuracy_score(test_labels, predicted_labels))

#Logistic Regression
lr_object=LogisticRegression()
lr_object.fit(train_features,train_labels)
predicted_labels=svc_object.predict(test_features)

st.write(classification_report(test_labels, predicted_labels))
st.write(confusion_matrix(test_labels, predicted_labels))
st.write(accuracy_score(test_labels, predicted_labels))
