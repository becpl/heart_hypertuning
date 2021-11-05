# Apply hyperparameter tuning (3 methods - grid, random and bayesian search) on SVM model
# for the dataset from assignment 8 and compare the best params and best score (make a word doc with table ).
# Submit code files as well as word document.

#import file and convert data
import pandas as pd
import pandas_profiling as pp
import numpy as np
df = pd.read_csv("heart.csv")

#check data import
print(df.columns)

#get information on dataset
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.info())
print(df.describe())
print(df.describe(include=['object']))

#run profile report to see what data needs cleaning
#profile = pp.ProfileReport(df)
#profile.to_file("heart_EdA.html")
# 5 categorical values need converting
#(Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope
# resting BP and cholesterol need 0 values converting

#tidy data up
#calculate mean values for resting BP and cholesterol
mean_Cholesterol = df["Cholesterol"].mean()
mean_RestingBP = df["RestingBP"].mean()
print("\n\n--------------------------------------------------------------------------")
print("average Cholesterol ",mean_Cholesterol)
print("average RestingBP ",mean_RestingBP)
#replcae 0 values in resting BP and cholesterol to mean values
df["Cholesterol"] = df["Cholesterol"].replace({0:mean_Cholesterol})
df["RestingBP"] = df["RestingBP"].replace({0:mean_RestingBP})
print(df.info())
#convert objects to numerical values for data modeling
#start with sex - only 2 possible values, can be handled with data replacement
#assign M to 1 and F to 2
df["ExerciseAngina"] = df["ExerciseAngina"].replace({"Y":"1"})
df["ExerciseAngina"] = df["ExerciseAngina"].replace({"N":"0"})
df["ExerciseAngina"] = df["ExerciseAngina"].astype(int)
df["Sex"] = df["Sex"].replace({"M":"1"})
df["Sex"] = df["Sex"].replace({"F":"2"})
df["Sex"] = df["Sex"].astype(int)

#check whether replacement worked
print(df.head())

#convert categorical data into numerical by creating dummy variables
df = pd.get_dummies(df,columns=["ChestPainType","RestingECG","ST_Slope"])
print("\n\n--------------------------------------------------------------------------")
print("after data conversion")
print(df.info())

#setup datasets for testing
y = df["HeartDisease"]
X = df.drop("HeartDisease",axis=1)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.30,random_state=0)
#Support Vector Machine modelling before hyperparameter tuning
from sklearn.svm import SVC
model_SVC = SVC(kernel="linear")
model_SVC.fit(X_train,y_train)
y_pred_SVC = model_SVC.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cnfn_SVC = confusion_matrix(y_test,y_pred_SVC)
cr_SVC = classification_report(y_test,y_pred_SVC)
acc_SVC = accuracy_score(y_test, y_pred_SVC)

print("\n\n--------------------------------------------------------------------------")
print("Support Vector Machine algorithm (Linear)")
print(cnfn_SVC)
print(cr_SVC)
print("Accuracy of SVM is: ",acc_SVC)
print("\n\n--------------------------------------------------------------------------")

#define a variable to enter the hyperparameters for tuning
param_space = {"kernel":["linear","rbf","sigmoid","poly"]}

#tuning technique = Random
tuning_random = RandomizedSearchCV(model_SVC, param_space,scoring="accuracy", cv=5, n_jobs=-1, random_state=0) #using randomized search
tuning_random.fit(X_train,y_train)
print("\n\n--------------------------------------------------------------------------")
print("Random Search")
print(tuning_random.best_params_)
print(tuning_random.best_score_)
print(tuning_random.best_estimator_)
print("\n\n--------------------------------------------------------------------------")#tuning=GridSearchCV(clf,param_space,scoring="accuracy",cv=5,N-jobs=-1,verbose=True)

#tuning technique = Grid
tuning_grid=GridSearchCV(model_SVC,param_space,scoring="accuracy",cv=5,n_jobs=-1,verbose=True)
tuning_grid.fit(X_train,y_train)
print("\n\n--------------------------------------------------------------------------")
print("Grid Search")
print(tuning_grid.best_params_)
print(tuning_grid.best_score_)
print(tuning_grid.best_estimator_)
print("\n\n--------------------------------------------------------------------------")

#tuning technique = Bayes
tuning_bayes=GridSearchCV(model_SVC,param_space,scoring="accuracy",cv=5,n_jobs=-1,verbose=True)
tuning_bayes.fit(X_train,y_train)
print("\n\n--------------------------------------------------------------------------")
print("Bayes Search")
print(tuning_bayes.best_params_)
print(tuning_bayes.best_score_)
print(tuning_bayes.best_estimator_)
print("\n\n--------------------------------------------------------------------------")
