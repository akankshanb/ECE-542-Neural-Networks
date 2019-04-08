#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rnn1 import generate_features
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection


# In[2]:


def dataextract(folder):
    dataset_arm = pd.read_csv(folder + "armIMU.txt", sep="  ", header=None, engine='python')
    dataset_wrist = pd.read_csv(folder + 'wristIMU.txt', sep="  ", header=None, engine='python')
    label = pd.read_csv(folder + 'detection.txt', sep="  ", header=None, engine='python')
    dataset_arm.columns = ["aa1", "aa2", "aa3", "ag1", "ag2", "ag3"]
    dataset_wrist.columns = ["wa1", "wa2", "wa3", "wg1", "wg2", "wg3"]
    label.columns = ["label"]
    dataset = pd.concat([dataset_arm, dataset_wrist], axis=1, sort=False)
    return dataset, label

def get_train_features(X):
    xyz = []
    for i in range(0, len(X), 40): # 70 percent overlap
        p = X.loc[i:i+150]
        set1 = p.loc[:,["aa1", "aa2", "aa3"]]
        set2 = p.loc[:,["ag1", "ag2", "ag3"]]
        set3 = p.loc[:,["wa1", "wa2", "wa3"]]
        set4 = p.loc[:,["wg1", "wg2", "wg3"]]
        f1 = generate_features(set1)
        f2 = generate_features(set2)
        f3 = generate_features(set3)
        f4 = generate_features(set4)
        f = np.concatenate((f3, f4), axis=0)
        f = np.concatenate((f1, f2, f), axis=0)
        features=f.tolist()
        xyz.append(features)
    return(pd.DataFrame(xyz))

def get_test_features(X):
    xyz = []
    for i in range(0, len(X)):
        p = X.loc[i:i+150]
        set1 = p.loc[:,["aa1", "aa2", "aa3"]]
        set2 = p.loc[:,["ag1", "ag2", "ag3"]]
        set3 = p.loc[:,["wa1", "wa2", "wa3"]]
        set4 = p.loc[:,["wg1", "wg2", "wg3"]]
        f1 = generate_features(set1)
        f2 = generate_features(set2)
        f3 = generate_features(set3)
        f4 = generate_features(set4)
        f = np.concatenate((f3, f4), axis=0)
        f = np.concatenate((f1, f2, f), axis=0)
        features=f.tolist()
        xyz.append(features)
    return(pd.DataFrame(xyz))


# In[3]:


train_folder1 = '/Users/akankshabhattacharyya/Downloads/NCSU/SEM2 Courses/Neural Networks/Project3/Training Data/Session01/'
train_folder2 = '/Users/akankshabhattacharyya/Downloads/NCSU/SEM2 Courses/Neural Networks/Project3/Training Data/Session05/'
train_folder3 = '/Users/akankshabhattacharyya/Downloads/NCSU/SEM2 Courses/Neural Networks/Project3/Training Data/Session06/'
train_folder4 = '/Users/akankshabhattacharyya/Downloads/NCSU/SEM2 Courses/Neural Networks/Project3/Training Data/Session07/'
train_folder5 ='/Users/akankshabhattacharyya/Downloads/NCSU/SEM2 Courses/Neural Networks/Project3/Training Data/Session12/'
train_folder6 = '/Users/akankshabhattacharyya/Downloads/NCSU/SEM2 Courses/Neural Networks/Project3/Training Data/Session13/'

Xtrain1, Ytrain1 = dataextract(train_folder1)
Xtrain2, Ytrain2 = dataextract(train_folder2)
Xtrain3, Ytrain3 = dataextract(train_folder3)
Xtrain4, Ytrain4 = dataextract(train_folder4)
Xtrain5, Ytrain5 = dataextract(train_folder5)
Xtrain6, Ytrain6 = dataextract(train_folder6)

Xtrain, Ytrain = pd.concat([Xtrain1, Xtrain2, Xtrain3, Xtrain4, Xtrain5, Xtrain6], ignore_index=True), pd.concat([Ytrain1, Ytrain2, Ytrain3, Ytrain4, Ytrain5, Ytrain6], ignore_index=True)
Xf = get_train_features(Xtrain)
Xf = pd.DataFrame(preprocessing.normalize(np.array(Xf)))
Yf = pd.DataFrame([1 if any(Ytrain.iloc[i:i + 150, 0]) else 0 for i in range(0, len(Ytrain), 40)])
Yf = np.ravel(Yf)


# In[5]:


def rf_param_selection(X, y):
    n_estimators = [200,300,400,500]
    max_depth = [7,8,9]
    max_features = ['sqrt','log2']
    param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
              }
    grid_search = model_selection.GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

    grid_search.fit(X, y)
    return grid_search.best_params_


# In[6]:


print("Training")
rf = RandomForestClassifier()
best = rf_param_selection(Xf, Yf)
print(best)


# In[7]:


rf1 = RandomForestClassifier(n_estimators= best['n_estimators'], max_features = best['max_features'], max_depth=best['max_depth'])
rf1.fit(Xf, Yf)


# In[ ]:


test_folder1 = '/Users/akankshabhattacharyya/Downloads/NCSU/SEM2 Courses/Neural Networks/Project3/Test Data 1/Session02/'
print("Testing")
Xtest, Ytest = dataextract(test_folder1)
Xtf = get_test_features(Xtest)
Xtf = Xtf.fillna(0)
Xtf = pd.DataFrame(preprocessing.normalize(np.array(Xtf)))
predictions = rf1.predict(Xtf)
predicitons.to_csv()


# In[ ]:




