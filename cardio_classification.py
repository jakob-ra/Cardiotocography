# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:07:01 2020

@author: Jakob
"""

##############################################################################################################
# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import imblearn
import copy
import scipy
from sklearn.impute import KNNImputer

##############################################################################################################
# Data

# Read data
url = "https://raw.githubusercontent.com/jakob-ra/Cardiotocography/master/cardiotocography.csv"
df = pd.read_csv(url, sep=';', decimal=',')

# Load variable names
url = "https://raw.githubusercontent.com/jakob-ra/Cardiotocography/master/varnames.csv"
varnames = pd.read_csv(url, sep=';')
cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'mSTV', 'ALTV', 'mLTV', 'Width', 'Min',
'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'NSP']
varnames = varnames.set_index('var').T[cols].T
# varnames = varnames.set_index('var').T.to_dict(orient='records')

df.dtypes  # check that all dtypes are float

df.isnull().sum().sum()  # check: number of missing values is 0

# use subset of columns
cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min',
'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'NSP']
df = df[cols]

## Hist for dependent variable (fetal status)
# fig, ax = plt.subplots(1, 1, figsize = (5,5))
# df.NSP.hist(ax=ax)
# ax.set_xticks([1,2,3])
# ax.set_xticklabels(['Normal', 'Suspect', 'Pathologic'])
# plt.show()

# Make binary outcome variable (normal,suspect+pathological)
df['status'] = np.where(df.NSP == 1, 0, 1)

# Histogram for all variables
# fig, ax = plt.subplots(1, 1, figsize = (15,20))
# df.hist(ax=ax)
# plt.show()
#
# Boxplot for scaled X variables
# fig, ax = plt.subplots(1, 1, figsize = (15,15))
# df_scale_X = pd.DataFrame(sklearn.preprocessing.scale(df),
#                          columns=df.columns).drop(columns=['NSP','status'])
# df_scale_X.boxplot(ax=ax, rot=45)
# plt.show()

# shuffles to break up any inherent order the observations might have (important for k-fold crossvalidation)
df = df.sample(frac=1, random_state=0)  # fix seed to make results reproducible

# make vector of class labels and feature matrix
y, X = df.status.values, df.drop(columns=['NSP', 'status']).values.astype('float')

##############################################################################################################
# Feature engineering


def lof_outlier_removal(X, share_out=10**(-20)):
    """ Removes outliers from X using local outlier factors. Number of removed
    outliers based on share_out (= prior on / desired share of outliers). len(X_transform)<len(X_original)
    """
    new_X = copy.deepcopy(X)
    lof = sklearn.neighbors.LocalOutlierFactor(n_neighbors=50, algorithm='auto',
                                               leaf_size=30, metric='minkowski', p=2, metric_params=None,
                                               contamination=share_out, novelty=False, n_jobs=None)

    outliers = lof.fit_predict(X)
    outlier_indices = [i for i in range(len(X)) if outliers[i] == -1]
    for outlier in outlier_indices:
        new_X = np.delete(X, outlier, axis=0)

    return new_X

# Make lof outlier_removal a transformer function
lof_outlier_removal = sklearn.preprocessing.FunctionTransformer(lof_outlier_removal)


def zscore_outlier_removal(X, threshold=5):
    """ Sets feature values in X that are more than threshold times standard deviation away from their mean
    to NaN. Returns X with original length but some column values are NaN.
    """
    new_X = copy.deepcopy(X)

    new_X[abs (sklearn.preprocessing.scale (X)) > threshold] = np.nan

    return new_X

# Make zscore feature outlier removal a transformer function
zscore_outlier_removal = sklearn.preprocessing.FunctionTransformer(zscore_outlier_removal,
                                                                   kw_args=dict(threshold=5))

# Impute missing (outliers) via KNN
KNN_impute = KNNImputer()

# add feature polynomials up to degree d
poly = sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

# demean and scale to unit variance
scale = sklearn.preprocessing.StandardScaler()

# Put all processing and feature engineering in a pipeline 
processing = imblearn.pipeline.Pipeline([('outlier', zscore_outlier_removal), ('impute', KNN_impute),
                                         ('poly', poly), ('scale', scale)], verbose=False)

##############################################################################################################
# Functions

def k_fold_strat_oversamp(X, y, k=10):
    """ Returns k bins with training and test samples where each bin has the original
    class distribution (via stratisfied sampling) and for the training sets
    we then oversample the underrepresented class with naive random oversampling
    to get balance.
    """
    k_folds = sklearn.model_selection.StratifiedKFold(n_splits=k)
    cv_indices = []
    for train_idx, test_idx, in k_folds.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]

        # Random oversampling
        ros = imblearn.over_sampling.RandomOverSampler(random_state=0)
        ros.fit_resample(X_train, y_train)
        train_idx = ros.sample_indices_

        cv_indices.append((train_idx, test_idx))

    #    plt.hist(y_train) # each training sample now has class balance

    return cv_indices

##############################################################################################################
# Logistic regression

# 10-fold cross-validation to find optimal penalty and l1_ratio,
# using balanced accuracy score and stratified, oversampled bins
log_reg_cv = sklearn.linear_model.LogisticRegressionCV(Cs=10, fit_intercept=True,
                                                   cv=k_fold_strat_oversamp(X, y, k=10), dual=False,
                                                   penalty='elasticnet',
                                                   scoring='balanced_accuracy', solver='saga', tol=0.0001,
                                                   max_iter=10000,
                                                   class_weight=None, n_jobs=-1, verbose=0, refit=True,
                                                   l1_ratios=[0, 0.1, 0.2, 0.5, 0.9, 0.95, 0.99, 1])

log_reg = sklearn.linear_model.LogisticRegression(fit_intercept=True, dual=False, C=0.3, l1_ratio=0.9,
                                                   penalty='elasticnet', solver='saga', tol=0.0001,
                                                   max_iter=10000, class_weight=None, n_jobs=1,
                                                  verbose=0)

log = copy.deepcopy(processing)
log.steps.append(['log_reg', log_reg])

param_grid = {"outlier__kw_args": [dict(threshold=i) for i in [100]],
              "poly__degree": [2],
              "log_reg__C": [np.e],
              "log_reg__l1_ratio": [0.99],
              "log_reg__class_weight": [{0: x, 1: 1-x} for x in np.linspace(0,1,11)]}

# weights to try
weights = np.linspace(0,1,11)
weight_grid = {"log_reg__class_weight": [{0: x, 1: 1-x} for x in weights]}

# penalty values to try
c_vals = np.linspace(0.5,5,11)
c_vals = [np.exp(i) for i in np.linspace(-3,3,10)]
c_grid = {"log_reg__C": c_vals}

cv = sklearn.model_selection.GridSearchCV(log, c_grid, scoring='balanced_accuracy', n_jobs=6,
                                                    refit=True, verbose=True, return_train_score=True)
cv.fit(X,y)

# cv.best_score_
# cv.best_params_
# cv.cv_results_

cv_results = pd.DataFrame(cv.cv_results_)

# Plot penalty values vs balanced accuracy score
plt.plot(np.log(c_vals), cv_results.mean_test_score, color="black")
plt.fill_between(c_vals, cv_results.mean_test_score+cv_results.std_test_score,
                 cv_results.mean_test_score-cv_results.std_test_score, color="gray")

plt.plot(np.log(c_vals), cv_results.mean_train_score, color="blue")
plt.fill_between(c_vals, cv_results.mean_train_score+cv_results.std_train_score,
                 cv_results.mean_train_score-cv_results.std_train_score, color="gray")
plt.show()

# Plot weights vs balanced accuracy score
# plt.plot(weights, weight_score.mean_score, color="black")
# plt.fill_between(weights, weight_score.mean_score+weight_score.std_score,
#                  weight_score.mean_score-weight_score.std_score, color="gray")
# plt.show()


# log_reg_cv.fit(X,y)
# log_reg_cv.scores_
# log_reg_cv.l1_ratio_
# log_reg_cv.C_
# log_reg_cv.coef_


# predicted = sklearn.model_selection.cross_val_predict(log_reg_cv, X, y, cv=k_fold_strat_oversamp(X,y,k=10))

# for i in range(len(log_reg_cv.coef_[0])):
#     print('{}: {}'.format(varnames.label.iloc[i], log_reg_cv.coef_[0][i]))

# sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None,
#    n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, 
#    pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)

# pred = clf.predict(X)
# sklearn.metrics.confusion_matrix(y, pred)
# sklearn.metrics.balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
