import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

#here collection data csv file
df = pd.read_csv('diabetes_data_upload.csv')

# statistics decriptive 
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df.columns)

'''
This algrithm for classificatio problem
---------------------------------------
k-Nearest Neighbors
Naive Bayes classifiers
Support Vector Machines
Decision Trees
Random Forests
Neural Networks
'''
       
#f Feature transformation to convert str to num    
lab = LabelEncoder()
df[['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity', 'class']]=df[['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity', 'class']].apply(lab.fit_transform)
   
       
       
print(df.head())       

x,y = df.drop('class', axis=1).values, df['class'].values

#scaler
std = StandardScaler()
x_std = std.fit_transform(x)

#Split dataset
xtrain, xtest ,ytrain, ytest = train_test_split(x_std,y, test_size=0.25)

#Model 
log = LogisticRegression(solver='liblinear',C=10.0)
log.fit(xtrain, ytrain)
pred = log.predict(xtest)

#here create dataframe to see good between each other Prodict v and Actual v
predict_dataframe = pd.DataFrame({"predict":pred,"Actual":ytest})
print(predict_dataframe)

#Statistics probability %%
prod_proba = log.predict_proba(xtest)
pred_p = pd.DataFrame(prod_proba)
print(pred_p)

# here how many correct values  in actual predict values  compare with actual v;
correct = (pred == ytest).sum()
incorrect = (pred != ytest).sum()

print(f"correct values in model:{100 * correct / len(pred) :.2f}")
print(log.classes_)
print(f"Incorrect {correct}")
print(f"Incorrect: {incorrect}")
print(classification_report(ytest, pred))





