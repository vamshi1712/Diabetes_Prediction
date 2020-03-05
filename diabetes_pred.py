# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 23:22:41 2020

@author: vamshi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('C:/Users/vamshi/Desktop/diabetes/diabetes.csv')

dataset.shape

dataset.columns


print("# rows in dataframe {0}".format(len(dataset)))
print("# rows missing Glucose: {0}".format(len(dataset.loc[dataset['Glucose'] == 0])))
print("# rows missing BloodPressure: {0}".format(len(dataset.loc[dataset['BloodPressure'] == 0])))
print("# rows missing SkinThickness: {0}".format(len(dataset.loc[dataset['SkinThickness'] == 0])))
print("# rows missing insulin: {0}".format(len(dataset.loc[dataset['Insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(dataset.loc[dataset['BMI'] == 0])))
print("# rows missing DiabetesPedigreeFunction: {0}".format(len(dataset.loc[dataset['DiabetesPedigreeFunction'] == 0])))
print("# rows missing age: {0}".format(len(dataset.loc[dataset['Age'] == 0])))

      
ds = dataset

dataset.corr()

sns.heatmap(dataset.corr())
      
#filling the missing data

#replacing 0 with NaN
dataset['BloodPressure'] = dataset['BloodPressure'].replace(0,np.NaN)
dataset['SkinThickness'] = dataset['SkinThickness'].replace(0,np.NaN)
dataset['Insulin'] = dataset['Insulin'].replace(0,np.NaN)
dataset['BMI'] = dataset['BMI'].replace(0,np.NaN)

dataset['BloodPressure'].isnull().sum()

#glucose

print("# rows missing Glucose: {0}".format(len(dataset.loc[dataset['Glucose'] == 0])))

dataset['Glucose'].mode()
#substituting the mode in missing place
dataset.loc[dataset['Glucose']==0,'Glucose' ] = 99



#Blood pressure


dataset['BloodPressure'].fillna(method='bfill',inplace=True)
dataset['BloodPressure'].isnull().sum()


#skinthickness

dataset['SkinThickness'].isnull().sum()

dataset['SkinThickness'].fillna(method='bfill',limit = 113,inplace=True)
dataset['SkinThickness'].fillna(method='ffill',inplace=True)




#Insulin

dataset['Insulin'].isnull().sum()


dataset['Insulin'].fillna(method='bfill',inplace=True)

#BMI

dataset['BMI'].isnull().sum()

dataset['BMI'].fillna(method='bfill',inplace=True)

dataset['BMI'].fillna(method='ffill',inplace=True)

dataset['BMI'] = dataset['BMI'].round()

#DiabetesPedigreeFunction

dataset['DiabetesPedigreeFunction'] = round(dataset['DiabetesPedigreeFunction'],1)


dataset.isnull().sum()

#data is clean 

from sklearn.model_selection import train_test_split

feature_col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class_names = ['Outcome']


X = dataset[feature_col_names].values # these are factors for the prediction
y = dataset[predicted_class_names].values # this is what we want to predict

split_test_size = 0.3

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = split_test_size,random_state=42)


#naivebayes

from sklearn.naive_bayes import GaussianNB

# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()

nb_model.fit(X_train, y_train.ravel())

# predict values using training data
nb_predict_train = nb_model.predict(X_train)

# import the performance metrics library from scikit learn
from sklearn import metrics

# check naive bayes model's accuracy
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,nb_predict_train)))
print()


nb_predict_test=nb_model.predict(X_test)


print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,nb_predict_test)))



#randomforest


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42) 
rf_model.fit(X_train,y_train.ravel())

rf_predict_train = rf_model.predict(X_train)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,rf_predict_train)))
print()

rf_predict_test = rf_model.predict(X_test)
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,rf_predict_test)))
print()

print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, rf_predict_test) )
print("")

print("Classification Report")
print(metrics.classification_report(y_test, rf_predict_test))


#decision tree
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4,random_state=0)
tree.fit(X_train,y_train)
#print("Accuracy on training set: {:.3f}".format(tree.score(X_train,y_train)))
#print("Accuracy on test set: {:.3f}".format(tree.score(X_test,y_test)))

dt_predict_train = tree.predict(X_train)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,dt_predict_train)))
print()

dt_predict_test = tree.predict(X_test)
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,dt_predict_test)))
print()

print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, dt_predict_test) )
print("")

print("Classification Report")
print(metrics.classification_report(y_test, dt_predict_test))


#svm type1

from sklearn import svm
machine1 = svm.SVC(kernel = 'linear')
machine1.fit(X_train,y_train)

sv1_pred_train = machine1.predict(X_train)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,sv1_pred_train)))
print()

sv1_pred_test = machine1.predict(X_test)
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,sv1_pred_test)))
print()

print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, sv1_pred_test) )
print("")

print("Classification Report")
print(metrics.classification_report(y_test, sv1_pred_test))

#svm type2

machine2 = svm.SVC(kernel = 'rbf')
machine2.fit(X_train,y_train)

sv2_pred_train = machine2.predict(X_train)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,sv2_pred_train)))
print()

sv2_pred_test = machine2.predict(X_test)
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,sv2_pred_test)))
print()

print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, sv2_pred_test) )
print("")

print("Classification Report")
print(metrics.classification_report(y_test, sv2_pred_test))

#svm type3

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
pipe_svc = Pipeline([('scl', StandardScaler()),('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},{'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=-1)
gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)


clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))

sv3_pred_train = clf.predict(X_train)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,sv3_pred_train)))
print()

sv3_pred_test = clf.predict(X_test)
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,sv3_pred_test)))
print()

print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, sv3_pred_test) )
print("")

print("Classification Report")
print(metrics.classification_report(y_test, sv3_pred_test))



#ANN

from sklearn.neural_network import MLPClassifier
# Create a RandomForestClassifier object
ann_model = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500, random_state=42)

ann_model.fit(X_train, y_train.ravel())

prediction_from_trained_data = ann_model.predict(X_train)


ann_accuracy_traindata = metrics.accuracy_score(y_train, prediction_from_trained_data)

print ("Accuracy of our ANN model is : {0:.4f}".format(ann_accuracy_traindata))

ann_predict_test = ann_model.predict(X_test)

#get accuracy
ann_accuracy_testdata = metrics.accuracy_score(y_test, ann_predict_test)

#print accuracy
print ("Accuracy: {0:.4f}".format(ann_accuracy_testdata))












