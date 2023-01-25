import pandas as pd
from pandas import Series, DataFrame
#Packages related to data visualizaiton
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
#Unbalanced dataset
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
#SVM to measure outliers
from sklearn.svm import OneClassSVM
from pylab import rcParams
from xgboost import XGBClassifier
import csv
import os
import joblib

data=pd.read_csv("card_transdata.csv")
Total_transactions = len(data)
normal = len(data[data.fraud == 0])
fraudulent = len(data[data.fraud == 1])
fraud_percentage = round(fraudulent/normal*100, 2)
# print('Total number of Trnsactions are {}'.format(Total_transactions))
# print('Number of Normal Transactions are {}'.format(normal))
# print('Number of fraudulent Transactions are {}'.format(fraudulent))
# print('Percentage of fraud Transactions is {}'.format(fraud_percentage))

X = data.iloc [:, :-1]
Y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.30, random_state = 1)

param = {'learning_rate': 0.2,
          'max_depth': 2, 
          'n_estimators':200,
          'subsample':0.9,
         'objective':'binary:logistic'}
xgb_model = XGBClassifier()
xgb = xgb_model.fit(X_train, y_train)
# y_test_pred_proba = xgb.predict_proba(X_test)[:,1]
# roc_auc = metrics.roc_auc_score(y_test, y_test_pred_proba)
xgb_scores = cross_val_score(xgb,X_train,y_train,scoring = 'r2',cv = 5)
xgb_scores
xgb_pred = xgb.predict(X_test)
xgb_TrainAccuracy =  accuracy_score(y_train, xgb.predict(X_train))
xgb_TestAccuracy =  accuracy_score(y_test, xgb_pred)
xgb_ConfusionMatrix =  confusion_matrix(y_test, xgb_pred)
# print("Accuracy of validation data is {}".format(xgb_scores.mean()))
# print('Accuracy score of the train data is {}'.format(xgb_TrainAccuracy))
# print('Accuracy score of the test data is {}'.format(xgb_TestAccuracy))
# print('Confusion Matrix - {}'.format(xgb_ConfusionMatrix))

print("Enter Distance from home: ")
v1 = float(input())
print("Enter Distance from last transaction: ")
v2 = float(input())
print("Enter Ratio to median purchase price: ")
v3 = float(input())
print("Was it the same retailer as the last one? 1.Yes 2.No: ")
v4 = int(input())
print("Did you use chip? 1.Yes 2.No: ")
x = int(input())
v5 = 1 if x == 1 else 0
print("Did you use your pin number? 1.Yes 2.No: ")
x = int(input())
v6 = 1 if x == 1 else 0
print("Is it an online order? 1.Yes 2.No: ")
x = int(input())
v7 = 1 if x == 1 else 0

f = open("inpdata.csv","a",newline="")
tup0 = ("distance_from_home","distance_from_last_transaction","ratio_to_median_purchase_price","repeat_retailer",
"used_chip","used_pin_number","online_order")
tup1 = (v1,v2,v3,v4,v5,v6,v7)
writer = csv.writer(f)
writer.writerow(tup0)
writer.writerow(tup1)
f.close()
print("Input successfull.....")

joblib.dump(xgb_model,"credit_card_model")
x_model = joblib.load("credit_card_model")
inpdata = pd.read_csv("inpdata.csv")
output = x_model.predict(inpdata)
if output == 0:
    print("The transaction is not fraudulent!")
else:
    print("The transaction is detected as fraudulent!")
os.remove("inpdata.csv")