# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:07:01 2020

@author: Jakob
"""

###########################################################
### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


###########################################################
### Data

# Read data
url = 'https://raw.githubusercontent.com/jakob-ra/Cardiotocography/master/cardiotocography.csv'
df = pd.read_csv(url, sep=';', decimal=',')

# Load variable names
url = 'https://raw.githubusercontent.com/jakob-ra/Cardiotocography/master/varnames.csv'
varnames = pd.read_csv(url, sep=';')


df.dtypes # check that all dtypes are float

df.isnull().sum().sum() # check: number of missing values is 0

# Histogram for all predictors
fig, ax = plt.subplots(1, 1, figsize = (15,20))
df.drop(columns='NSP').hist(ax=ax)
plt.show()

# Hist for dependent variable (fetal status)
fig, ax = plt.subplots(1, 1, figsize = (5,5))
df.NSP.hist(ax=ax)
ax.set_xticks([1,2,3])
ax.set_xticklabels(['Normal', 'Suspect', 'Pathologic'])
plt.show()

# Make binary outcome variable (normal,suspect+pathological)
df['status'] = np.where(df.NSP==1, 0, 1)

###########################################################
### Logistic regression
y, X = df.status, df.drop(columns=['NSP','status']) # make regressors
y, X = sklearn.preprocessing.scale(y),sklearn.preprocessing.scale(X) #scale

clf = sklearn.linear_model.LogisticRegression(penalty='elasticnet', 
    max_iter=10000, class_weight='balanced', solver='saga', random_state=0, 
    n_jobs=-1, c=0.5, l1_ratio=0.9)
clf.fit(X, y)
clf.score(X, y)
pred = clf.predict(X)

sklearn.metrics.confusion_matrix(y, pred)
sklearn.metrics.balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
###########################################################
### Functions
def fun(arg):
    """Purpose: 
    Inputs: 
    Returns: 
    """
    return arg


