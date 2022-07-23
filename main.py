import pandas as pd
import streamlit as st


st.header("Customer Banking Churn App")

data =pd.read_csv('https://raw.githubusercontent.com/regan-mu/ADS-April-2022/main/Assignments/Assignment%202/banking_churn.csv')
#data.head()
#data.describe(include='all')

data.info()
#data.shape
print("Number of rows", data.shape[0])
print("Number of columns", data.shape[1])

data.isnull().sum()

#data.columns

data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

data.head()
data['Geography'].unique()
data['Gender'].unique()

data = pd.get_dummies(data, drop_first=True)

data.head()
data['Exited'].value_counts()
X = data.drop('Exited', axis=1)
y = data['Exited']

from imblearn.over_sampling import SMOTE
X_res,y_res=SMOTE().fit_resample(X,y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_res, y_res, test_size=0.2, random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#X_train
#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
print(lr.fit(X_train, y_train))
y_pred1=lr.predict(X_test)
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score
print(accuracy_score(y_test,y_pred1))
print(precision_score(y_test,y_pred1))
print(recall_score(y_test,y_pred1))
print(recall_score(y_test,y_pred1))

#SVC
from sklearn import svm
svm= svm.SVC()
print(svm.fit(X_train, y_train))
y_pred2=svm.predict(X_test)
y_pred2=svm.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
print(f1_score(y_test,y_pred2))
print(recall_score(y_test,y_pred2))

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
print(knn.fit(X_train, y_train))
y_pred3=knn.predict(X_test)
y_pred3=knn.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
print(recall_score(y_test,y_pred3))
print(f1_score(y_test,y_pred3))

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
print(dt.fit(X_train, y_train))
y_pred4=dt.predict(X_test)
print(dt.fit(X_train, y_train))
y_pred4=dt.predict(X_test)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
print(rf.fit(X_train,y_train))
y_pred5=rf.predict(X_test)
print(accuracy_score(y_test,y_pred5))
print(precision_score(y_test,y_pred5))
print(recall_score(y_test,y_pred5))
print(f1_score(y_test,y_pred5))

#Gradient Boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
print(gbc.fit(X_train,y_train))
y_pred6=gbc.predict(X_test)
print(accuracy_score(y_test,y_pred6))
print(precision_score(y_test,y_pred6))
print(recall_score(y_test,y_pred6))
print(f1_score(y_test,y_pred6))

final_data= pd.DataFrame({'Models':["LR","SVC","KNN","DT","RF","GBC"],
                         "ACC":[accuracy_score(y_test,y_pred1),
                               accuracy_score(y_test,y_pred2),
                               accuracy_score(y_test,y_pred3),
                               accuracy_score(y_test,y_pred4),
                               accuracy_score(y_test,y_pred5),
                               accuracy_score(y_test,y_pred6)]})
# final_data

X_res=sc.fit_transform(X_res)
model=rf.fit(X_res,y_res)

import pickle
import streamlit as st
pickle_out=open("classifier.pkl",mode="wb")
pickle.dump(model,pickle_out)
pickle_out.close()

pickle_in=open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)
model.predict([[619, 42, 2, 0.0, 0, 0, 0, 101348.88, 0, 0, 0]])

#data.columns
@st.cache()
# defining the function which will make the prediction using the data which the user inputs
def prediction(CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
              Geography_Germany, Geography_Spain):

      # Pre-processing user input
      if Gender == "Male":
            Gender = 0
      else:
            Gender = 1
      if Geography_Spain == "Yes":
            Geography_Spain = 1
      else:
            Geography_Spain = 0

      if Geography_Germany == "Yes":
            Geography_Germany = 1
      else:
            Geography = 0

        # Making prediction
            prediction = classifier.predict(
                [[CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
                Geography_Germany, Geography_Spain]]

        if Gender == "male":
            Gender=1
        else:
            Gender=0
        if Geography_Germany == "Germany":

        if prediction == 0:
            pred = 'Not Exited!'
        else:
            pred = 'Exited!'
        return pred

import streamlit.components.v1 as components
import os

# this is the main function in which we define our webpage
# front end elements of the web page
class Geography_spain:
    pass


class Geography_Germany:
    pass


def main():
    html_temp=""" 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Bank Customer Churn Prediction ML App</h1> 
    </div> 
    """

    # display the front end aspect
    # st.markdown(html_temp, unsafe_allow_html = True)

    # following lines create boxes in which user can enter data required to make prediction
    Gender = st.selectbox("Customer's Gender", ("Male", "Female"))
    Age = st.text_input("Customer's Age")
    NumOfProducts = st.selectbox("Number of Bank Products", ("1", "2", "3", "4"))
    Tenure = st.selectbox("Tenure",
                          ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"))
    HasCrCard = st.selectbox('The Customer has a Credit Card?', ("Yes", "No"))
    IsActiveMember = st.selectbox('Is The Customer an Active Member?', ("Yes", "No"))
    EstimatedSalary = st.number_input("Estimated Salary")
    Balance = st.number_input("Account Balance")
    CreditScore = st.text_input("Credit Score")
    Geography = st.selectbox('Geography', ("Germany", "France", "Spain"))

    result = "  "

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
                EstimatedSalary, Geography_Germany, Geography_spain)
        st.success('The Customer will {}'.format(result))


if __name__ == '__main__':
    main()
