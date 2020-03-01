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
import seaborn as sns
import sklearn
from sklearn.inspection import permutation_importance

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

# Make binary outcome variable (normal,suspect+pathological)
df['status'] = np.where(df.NSP == 1, -1, 1)  # recodes normal to -1 and everything else to 1

# shuffle data once to break up any inherent order the observations might have (important for k-fold
# crossvalidation)
df = df.sample(frac=1, random_state=seed)

# make vector of class labels and feature matrix
df_X = df.drop(columns=['NSP', 'status'])
y, X = df.status.values, df_X.values.astype('float')

#############################################################################################################
# Descriptice statistics

# Hist for dependent variable (fetal status)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
df.NSP.hist(ax=ax)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Normal', 'Suspect', 'Pathologic'])
plt.show()

# Histogram for all variables
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
df.hist(ax=ax)
# plt.savefig('hist', dpi=150)
plt.show()


# Boxplots for feature distributions
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
df_scale_X = pd.DataFrame(sklearn.preprocessing.scale(df), columns=df.columns).drop(
    columns=['NSP', 'status'])
df_scale_X.boxplot(ax=ax, rot=45)
plt.savefig('boxplot', dpi=150)
plt.show()

# Feature correlation heatmap
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
corr = df_X.corr()
corr = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, ax=ax)
plt.xticks(rotation=45)
plt.show()

print(varnames.to_latex())
print(varnames.iloc[7])
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
    learning_rate=1.0, random_state=seed, algorithm='SAMME')  # SAMME is the algorithm I implemented

adaboost_m1_package.fit(X, y)
y_pred_package = adaboost_m1_package.predict(X)

# verify that the predictions are identical
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
lof_outlier_removal = imblearn.FunctionSampler(func=lof_outlier_removal, kw_args=dict(share_out=10 ** (-2)))


def zscore_outlier_removal(X, threshold=5):
    """ Sets feature values in X that are more than threshold times standard deviation away from their mean
    to NaN. Returns X with original length but some column values are NaN.
    """
    new_X = copy.deepcopy(X)
    new_X[abs(sklearn.preprocessing.scale(X)) > threshold] = np.nan

    return new_X


# Make zscore feature outlier removal a transformer function
zscore_outlier_removal = sklearn.preprocessing.FunctionTransformer(zscore_outlier_removal,
    kw_args=dict(threshold=5))

# Replace feature outliers with imputed values via KNN
KNN_impute = sklearn.impute.KNNImputer()

# SMOTE overssampling
smote = imblearn.over_sampling.SMOTE(sampling_strategy=1, random_state=seed)

# Polynomial feature expansion
poly = sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

# demean and scale to unit variance
scale = sklearn.preprocessing.StandardScaler()

# processing = imblearn.pipeline.Pipeline(
#     [('outlier', zscore_outlier_removal), ('impute', KNN_impute), ('poly', poly), ('scale', scale)],
#     verbose=False)

#############################################################################################################
# Logistic regression

log_reg = sklearn.linear_model.LogisticRegression(fit_intercept=True, dual=False, C=0.3, l1_ratio=0.9,
    penalty='elasticnet', solver='saga', tol=0.001, max_iter=5000, class_weight='balanced', n_jobs=1)

# Put all processing and feature engineering steps plus log_reg classifier in a pipeline

# lof observation outlier removal
log_lof = imblearn.pipeline.Pipeline(
    [('outlier', lof_outlier_removal), ('poly', poly), ('scale', scale), ('log_reg', log_reg)])

# zscore feature value outlier removal and subsequent imputation
log_z = sklearn.pipeline.Pipeline([('outlier', zscore_outlier_removal), ('impute', KNN_impute), ('poly',
    poly), ('scale', scale), ('log_reg', log_reg)])

# SMOTE oversampling pipeline (then have to set class weights in classifier to None rather than balanced
log_smote = imblearn.pipeline.Pipeline(
    [('smote', smote), ('poly', poly), ('scale', scale), ('log_reg', log_reg)])

# final pipeline
log = imblearn.pipeline.Pipeline(
    [('poly', poly), ('scale', scale), ('log_reg', log_reg)])

# values to try for cross-validation
l1_ratio_vals = [0, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
penalty_vals = [np.e ** i for i in np.linspace(-3, 3, 8)]
class_weights = np.linspace(0, 1, 11)
# [{-1: x, 1: 1 - x} for x in class_weights]}
lof_outlier_shares = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
poly_degrees = [1,2,3]
zscore_threshold_vals = [100, 7, 5, 3]
smote_minority_ratios = [0.3,0.5,0.7,0.8,0.9,0.95,1]

param_grid = {"outlier__kw_args": [dict(share_out=i) for i in [0]], "poly__degree": [2],
    "log_reg__C": [3.4], "log_reg__l1_ratio": [0.9],
    "log_reg__class_weight": ['balanced']}

param_grid_z = {"outlier__kw_args": [dict(threshold=i) for i in zscore_threshold_vals], "poly__degree":
    poly_degrees, "log_reg__C": [3.4], "log_reg__l1_ratio": [0.9]}

param_grid_smote = {"smote__sampling_strategy": smote_minority_ratios}

# stratified 5-fold cross-validation
cv = sklearn.model_selection.GridSearchCV(log_lof, param_grid, scoring='balanced_accuracy', n_jobs=8,
    refit=True, verbose=True, return_train_score=True, cv=5)
cv.fit(X, y)

print('\n Results for elastic net penalized logistic regression:')
print('The best out-of-sample performance is {}'.format(cv.best_score_))
print('Best parameter values: ', cv.best_params_)

# Permutation importance
result  = permutation_importance(cv.best_estimator_, X, y, scoring='balanced_accuracy',
    n_repeats=120, n_jobs=-1, random_state=seed)
perm_sorted_idx = result.importances_mean.argsort()

plt.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=df.columns[perm_sorted_idx])
fig.tight_layout()
plt.show()

# cv_z = sklearn.model_selection.GridSearchCV(log_z, param_grid_z, scoring='balanced_accuracy', n_jobs=-1,
#     refit=True, verbose=True, return_train_score=True, cv=5)
# cv_z.fit(X, y)
#
# print('The best out-of-sample performance is {}'.format(cv_z.best_score_))
# print('Best parameter values: ', cv_z.best_params_)

pred = cv.best_estimator_.predict(X)
print('In-sample confusion matrix of best estimator: ', sklearn.metrics.confusion_matrix(y, pred))

cross_val_pred = sklearn.model_selection.cross_val_predict(cv.best_estimator_, X, y, cv=5)
print('Out-of-sample confusion matrix of best estimator:\n{}'.format(sklearn.metrics.confusion_matrix(y,
    cross_val_pred)))


# Get validation curve (penalty values vs score)
penalty_vals = [np.e ** i for i in np.linspace(-6, 6, 32)] # make finer grid of penalty values
param_grid_learning = {"log_reg__C": penalty_vals}
cv = sklearn.model_selection.GridSearchCV(log, param_grid_learning, scoring='balanced_accuracy', n_jobs=8,
    refit=True, verbose=True, return_train_score=True, cv=5)
cv.fit(X, y) # rerun the cross-validation
cv_results = pd.DataFrame(cv.cv_results_)

# Plot the validation curve
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


# Construct learning curve
train_sizes = np.linspace(0.01,1,10)
learning_curve = sklearn.model_selection.learning_curve(cv.best_estimator_, X, y, train_sizes=train_sizes,
    cv=5, scoring='balanced_accuracy', exploit_incremental_learning=False, n_jobs=-1, verbose=1,
    random_state=seed, return_times=True)
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

# Plot
fig, axes = plt.subplots(2, figsize=(10, 8))
axes[0].set_xlabel("Training examples")
axes[0].set_ylabel("Score")

# Plot learning curve
axes[0].grid()
axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
    alpha=0.1, color="r")
axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
    alpha=0.1, color="g")
axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
axes[0].legend(loc="best")
axes[1].set_title("Learning curve")


# Plot n_samples vs fit_times
axes[1].grid()
axes[1].plot(train_sizes, fit_times_mean, 'o-')
axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
axes[1].set_xlabel("Training examples")
axes[1].set_ylabel("Fit time in seconds")
axes[1].set_title("Scalability of the model")

plt.show()

#############################################################################################################
# Support Vector Machine
svc = sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', class_weight='balanced',
    decision_function_shape='ovr')

svc_pipe = sklearn.pipeline.Pipeline([('scale', scale), ('svc', log_reg)])

param_grid = {"svc__C": penalty_vals, "svc__l1_ratio": l1_ratio_vals}

cv = sklearn.model_selection.GridSearchCV(svc_pipe, param_grid, scoring='balanced_accuracy', n_jobs=8,
    refit=True, verbose=True, return_train_score=True, cv=5)
cv.fit(X, y)

print('\n Results for support vector machine:')
print('The best out-of-sample performance is {}'.format(cv.best_score_)) #SVC worse
print('Best parameter values: ', cv.best_params_)

#############################################################################################################
# AdaBoost

tree = sklearn.tree.DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=seed)

boost = sklearn.ensemble.AdaBoostClassifier(base_estimator=tree, n_estimators=1000, learning_rate=1.0,
    random_state=seed, algorithm='SAMME.R')

# Put all processing and feature engineering in a pipeline
boost_pipe = imblearn.pipeline.Pipeline([('poly', poly), ('boost', boost)])

# values to try for cross-validation
learning_rates = [0.3,0.5,0.7,0.8,0.9,1]
n_iterations_vals = [200,500,1000,1500,2000]
tree_depths = [1,2,3,4,5]
poly_degrees = [1,2,3]

param_grid = {"poly__degree": [2], "boost__base_estimator__max_depth": [2], "boost__learning_rate": [0.7],
    "boost__n_estimators": [1500]}

# stratified 5-fold cross-validation
boost_cv = sklearn.model_selection.GridSearchCV(boost_pipe, param_grid, scoring='balanced_accuracy',
    n_jobs=8, refit=True, verbose=True, return_train_score=True, cv=5)
boost_cv.fit(X, y)

print('\n Results for elastic net penalized logistic regression:')
print('The best out-of-sample performance is {}'.format(boost_cv.best_score_))
print('Best parameter values: ', boost_cv.best_params_)

pred = cv.best_estimator_.predict(X)
print('In-sample confusion matrix of best estimator: ', sklearn.metrics.confusion_matrix(y, pred))

cross_val_pred = sklearn.model_selection.cross_val_predict(cv.best_estimator_, X, y, cv=5)
print('Out-of-sample confusion matrix of best estimator:\n{}'.format(sklearn.metrics.confusion_matrix(y,
    cross_val_pred)))

# Permutation importance
result  = permutation_importance(cv.best_estimator_, X, y, scoring='balanced_accuracy',
    n_repeats=120, n_jobs=-1, random_state=seed)
perm_sorted_idx = result.importances_mean.argsort()

plt.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=df.columns[perm_sorted_idx])
fig.tight_layout()
plt.show()

#############################################################################################################  # Manual k-fold + oversampling

# def k_fold_strat_oversamp(X, y, k=10):  #     """ Returns k bins with training and test samples where
# each bin has the original  #     class distribution (via stratisfied sampling) and for the training sets
#     we then oversample the underrepresented class with naive random oversampling  #     to get balance.
#     """  #     k_folds = sklearn.model_selection.StratifiedKFold(n_splits=k)  #     cv_indices = []  #
#     for train_idx, test_idx, in k_folds.split(X, y):  #         X_train, y_train = X[train_idx],
#     y[train_idx]  #  #         # Random oversampling  #         ros =
#     imblearn.over_sampling.RandomOverSampler(random_state=seed)  #         ros.fit_resample(X_train,
#     y_train)  #         train_idx = ros.sample_indices_  #  #         cv_indices.append((train_idx,
#     test_idx))  #  #     #    plt.hist(y_train) # each training sample now has class balance  #  #     return cv_indices

# Recall score
# sklearn.metrics.recall_score(y, pred)
