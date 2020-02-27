# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:07:01 2020

@author: Jakob
"""

##########################################################################################
# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import imblearn
import copy
import scipy
from sklearn.impute import KNNImputer
import sklearn

#############################################################################################################
# Data

# fix seed to make results reproducible
seed = 42

# Read data
url = "https://raw.githubusercontent.com/jakob-ra/Cardiotocography/master" \
      "/cardiotocography.csv"
df = pd.read_csv(url, sep=';', decimal=',')

# Load variable names
url = "https://raw.githubusercontent.com/jakob-ra/Cardiotocography/master/varnames.csv"
varnames = pd.read_csv(url, sep=';')
cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'mSTV', 'ALTV', 'mLTV', 'Width', 'Min', 'Max',
    'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'NSP']
varnames = varnames.set_index('var').T[cols].T
# varnames = varnames.set_index('var').T.to_dict(orient='records')

print(df.dtypes)  # check that all dtypes are float

print('Number of missing values: ', df.isnull().sum().sum())  # check for missing

# use subset of columns
cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max',
    'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'NSP']
df = df[cols]

# Hist for dependent variable (fetal status)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
df.NSP.hist(ax=ax)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Normal', 'Suspect', 'Pathologic'])
plt.show()

# Make binary outcome variable (normal,suspect+pathological)
df['status'] = np.where(df.NSP == 1, -1, 1)  # recodes normal to -1 and everything else to 1

# Histogram for all variables
fig, ax = plt.subplots(1, 1, figsize=(15, 20))
df.hist(ax=ax)
plt.show()

# Boxplot for scaled X variables
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
df_scale_X = pd.DataFrame(sklearn.preprocessing.scale(df), columns=df.columns).drop(
    columns=['NSP', 'status'])
df_scale_X.boxplot(ax=ax, rot=45)
plt.show()

# shuffle data once to break up any inherent order the observations might have (important for k-fold
# crossvalidation)
df = df.sample(frac=1, random_state=seed)

# make vector of class labels and feature matrix
y, X = df.status.values, df.drop(columns=['NSP', 'status']).values.astype('float')


#############################################################################################################
# Own implementation of AdaBoost algorithm

def adaboost_m1(X, y, clf, M=100):
    """ Implementation of the AdaBoost.M1 algorithm for binary classification. clf is the desired classifier
    and M is the number of iterations. Returns a vector of predictions.
    """
    # check classes
    if sum(~np.isin(y, [-1, 1])) != 0:
        raise Exception('The classes are not encoded as (-1,1) or there are more than two classes!')

    # initialize weights
    n_obs = len(y)
    obs_weights = np.array([1 / n_obs for i in range(n_obs)])  # start with equal weights

    # run loop
    y_pred_M = np.zeros((M, n_obs))
    alpha_M = np.zeros(M)
    for m in range(M):
        clf.fit(X, y, sample_weight=obs_weights)  # fit classifier using sample_weights
        y_pred_m = clf.predict(X)  # obtain predictions

        err_ind = np.array(y != y_pred_m).astype('int')  # vector indicating prediction errors
        err_m = obs_weights.T @ err_ind / np.sum(obs_weights)
        alpha_m = np.log((1 - err_m) / err_m)

        obs_weights = [obs_weights[i] * np.exp(alpha_m * err_ind[i]) for i in range(len(y))]
        obs_weights = obs_weights / np.sum(obs_weights)  # normalize weights to sum to 1

        # save predictions and alpha
        y_pred_M[m] = y_pred_m
        alpha_M[m] = alpha_m

    # Make vector of final predictions
    y_pred = [np.sign(np.sum([alpha_M[m] * y_pred_M[m, i] for m in range(M)])) for i in range(n_obs)]

    return y_pred


#############################################################################################################
# Verify that our own implementation gives the same results as the sklearn package

# Set classifier to a decision tree with maximum depth 2 (and fixed seed to select the
# same tree structure even if two splits would give the same improvement)
tree = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=seed)

# set number of iterations
M = 500

# get predictions from own implementation
y_pred = adaboost_m1(X, y, tree, M=M)

# get predictions from sklearn package
adaboost_m1_package = sklearn.ensemble.AdaBoostClassifier(base_estimator=tree, n_estimators=M,
    learning_rate=1.0, random_state=seed,
    algorithm='SAMME')  # SAMME is the algorithm described in the slides
adaboost_m1_package.fit(X, y)
y_pred_package = adaboost_m1_package.predict(X)

# verify that the results are identical
print('The results from sklearn and our own implementation are the same: ',
    np.sum(y_pred != y_pred_package) == 0)  # we get the same predictions

#############################################################################################################
# Feature engineering

# Outlier removal
def lof_outlier_removal(X, y, share_out=10 ** (-2)):
    """ Removes outlier observations from X using local outlier factors. Number of removed outliers based on
    share_out (prior on the share of outliers). Returns shortened sample X,y.
    """
    if share_out == 0:
        return X, y

    lof = sklearn.neighbors.LocalOutlierFactor(n_neighbors=50, algorithm='auto', leaf_size=30,
        metric='minkowski', p=2, metric_params=None, contamination=share_out, novelty=False, n_jobs=None)

    inlier = lof.fit_predict(X)

    return X[inlier == 1], y[inlier == 1]

# Make lof outlier_removal a transformer function
lof_outlier_removal = imblearn.FunctionSampler(func=lof_outlier_removal, kw_args=dict(share_out=10**(-2)))

def zscore_outlier_removal(X, threshold=5):
    """ Sets feature values in X that are more than threshold times standard deviation away from their mean
    to NaN. Returns X with original length but some column values are NaN.
    """
    new_X = copy.deepcopy(X)

    new_X[abs(sklearn.preprocessing.scale(X)) > threshold] = np.nan

    return new_X

# Make zscore feature outlier removal a transformer function
zscore_outlier_removal = sklearn.preprocessing.FunctionTransformer(zscore_outlier_removal, kw_args=dict(
    threshold=5))

# Replace feature outliers with imputed values via KNN
KNN_impute = KNNImputer()

# add feature polynomials up to degree d
poly = sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

# demean and scale to unit variance
scale = sklearn.preprocessing.StandardScaler()

# Put all processing and feature engineering in a pipeline
processing = imblearn.pipeline.Pipeline([('outlier', lof_outlier_removal), ('poly', poly), ('scale', scale)])

# processing = imblearn.pipeline.Pipeline(
#     [('outlier', zscore_outlier_removal), ('impute', KNN_impute), ('poly', poly), ('scale', scale)],
#     verbose=False)

#############################################################################################################
# Logistic regression

log_reg = sklearn.linear_model.LogisticRegression(fit_intercept=True, dual=False, C=0.3, l1_ratio=0.9,
    penalty='elasticnet', solver='saga', tol=0.001, max_iter=5000, class_weight='balanced', n_jobs=1)

log = copy.deepcopy(processing)
log.steps.append(['log_reg', log_reg])

# values to try for cross-validation
l1_ratio_vals = [0, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
penalty_vals = [np.e**i for i in np.linspace(-3,3,8)]
class_weights = np.linspace(0, 1, 11)
lof_outlier_shares = [0,0.01,0.02,0.03,0.05,0.1,0.15,0.2]
poly_degrees = [2]

param_grid = {"outlier__kw_args": [dict(share_out=i) for i in [0]],
    "poly__degree": poly_degrees,
    "log_reg__C": [3.4],
    "log_reg__l1_ratio": [0.9]}

# param_grid = {"outlier__kw_args": [dict(threshold=i) for i in [100]], "poly__degree": [2],
#     "log_reg__C": [np.e], "log_reg__l1_ratio": [0.99],
#     "log_reg__class_weight": [{0: x, 1: 1 - x} for x in class_weights}

# stratified 5-fold cross-validation
cv = sklearn.model_selection.GridSearchCV(log, param_grid, scoring='balanced_accuracy', n_jobs=8, refit=True,
    verbose=True, return_train_score=True, cv=5)
cv.fit(X, y)

print('The best out-of-sample performance is {}'.format(cv.best_score_))
print('Best parameter values: ', cv.best_params_)

pred = cv.best_estimator_.predict(X)
print('In-sample confusion matrix of best estimator: ', sklearn.metrics.confusion_matrix(y, pred))


# Plot penalty values vs balanced accuracy score
param_grid = {"log_reg__C": [np.e**i for i in np.linspace(-6,6,32)]}
cv.fit(X, y)
cv_results = pd.DataFrame(cv.cv_results_)

plt.plot(np.log(penalty_vals), cv_results.mean_test_score, color="black", label='Test')
plt.fill_between(np.log(penalty_vals), cv_results.mean_test_score + cv_results.std_test_score,
    cv_results.mean_test_score - cv_results.std_test_score, color="silver")

plt.plot(np.log(penalty_vals), cv_results.mean_train_score, color="blue", label='Train')
plt.fill_between(np.log(penalty_vals), cv_results.mean_train_score + cv_results.std_train_score,
    cv_results.mean_train_score - cv_results.std_train_score, color="bisque")
plt.axvline(np.log(cv.best_params_['log_reg__C']), c='r', linestyle='--')
plt.ylabel('Balanced accuracy')
plt.xlabel('Log of inverse penalty strength')
plt.title('Penalization vs performance for L1-ratio={}'.format(0.9))
plt.legend()
plt.show()

#############################################################################################################
# AdaBoost

# SMOTE
SMOTE = imblearn.over_sampling.SMOTE(sampling_strategy=1, random_state=seed)

boost_pipe.get_params()

tree = sklearn.tree.DecisionTreeClassifier(max_depth=2, random_state=seed)

boost = sklearn.ensemble.AdaBoostClassifier(base_estimator=tree, n_estimators=M, learning_rate=1.0,
    random_state=seed, algorithm='SAMME.R')

# Put all processing and feature engineering in a pipeline
boost_pipe = imblearn.pipeline.Pipeline([('outlier', lof_outlier_removal), ('oversample',
    SMOTE), ('poly', poly), ('scale', scale), ('boost', boost)])


param_grid = {"oversample__sampling_strategy": [0.5, 1],
    "poly__degree": [2,3],
    "boost__base_estimator__max_depth": [2],
    "boost__learning_rate": [0.7],
    "boost__n_estimators": [2000]}

# stratified 5-fold cross-validation
boost_cv = sklearn.model_selection.GridSearchCV(boost_pipe, param_grid, scoring='balanced_accuracy', n_jobs=8,
    refit=True, verbose=True, return_train_score=True, cv=5)
boost_cv.fit(X, y)

print('The best out-of-sample performance is {}'.format(cv.best_score_))
print('Best parameter values: ', cv.best_params_)

pred = cv.best_estimator_.predict(X)
print('In-sample confusion matrix of best estimator: ', sklearn.metrics.confusion_matrix(y, pred))


#############################################################################################################
# Manual k-fold + oversampling

# def k_fold_strat_oversamp(X, y, k=10):
#     """ Returns k bins with training and test samples where each bin has the original
#     class distribution (via stratisfied sampling) and for the training sets
#     we then oversample the underrepresented class with naive random oversampling
#     to get balance.
#     """
#     k_folds = sklearn.model_selection.StratifiedKFold(n_splits=k)
#     cv_indices = []
#     for train_idx, test_idx, in k_folds.split(X, y):
#         X_train, y_train = X[train_idx], y[train_idx]
#
#         # Random oversampling
#         ros = imblearn.over_sampling.RandomOverSampler(random_state=seed)
#         ros.fit_resample(X_train, y_train)
#         train_idx = ros.sample_indices_
#
#         cv_indices.append((train_idx, test_idx))
#
#     #    plt.hist(y_train) # each training sample now has class balance
#
#     return cv_indices
