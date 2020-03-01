# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:07:01 2020

@author: Jakob
"""

#####################################################################
# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imblearn
import copy
import seaborn as sns
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.impute import KNNImputer

#####################################################################
# Data

# fix seed to make results reproducible
seed = 42

# Read data
url = "https://raw.githubusercontent.com/jakob-ra/Cardiotocography" \
      "/master/cardiotocography.csv"
df = pd.read_csv(url, sep=';', decimal=',')

print(df.dtypes)  # check data types

print('Number of missing values: ',
    df.isnull().sum().sum())  # check for missing

# use subset of columns
cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV',
    'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode',
    'Mean', 'Median', 'Variance', 'Tendency', 'NSP']
df = df[cols]

# Make binary outcome variable (normal,suspect+pathological)
df['Status'] = np.where(df.NSP == 1, -1,
    1)  # recodes normal to -1 and everything else to 1

# shuffle data once to break up any inherent order the
# observations might have (important for k-fold
# crossvalidation)
df = df.sample(frac=1, random_state=seed)

# make vector of class labels and feature matrix
df_X = df.drop(columns=['NSP', 'Status'])
y, X = df.Status.values, df_X.values.astype('float')

#####################################################################
# Descriptive plots

# Histogram for all variables
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
df.hist(ax=ax)
plt.savefig('hist', dpi=150)
plt.show()

# Boxplots for feature distributions
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
df_scale_X = pd.DataFrame(sklearn.preprocessing.scale(df),
    columns=df.columns).drop(columns=['NSP', 'Status'])
df_scale_X.boxplot(ax=ax, rot=45)
plt.savefig('boxplot', dpi=150)
plt.show()

# Feature correlation heatmap
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
corr = df.drop(columns=['NSP']).corr()
corr = corr.round(decimals=2)
corr = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
    annot=True, ax=ax)
plt.xticks(rotation=45)
plt.savefig('corr', dpi=150)
plt.show()

# Plot example tree
plt.figure(figsize=(10, 15))
example_tree = sklearn.tree.DecisionTreeClassifier(max_depth=2)
example_tree.fit(X, y)
sklearn.tree.plot_tree(example_tree, feature_names=df_X.columns,
    class_names=['normal', 'suspect/pathological'], filled=True,
    node_ids=True, precision=2, impurity=True)
plt.tight_layout()
# plt.savefig('tree', dpi=150)
plt.show()


#####################################################################
# Own implementation of AdaBoost algorithm

def adaboost_m1(X, y, clf, M=500):
    """ Implementation of the AdaBoost.M1 algorithm for binary
    classification. clf is the desired classifier
    and M is the number of iterations. Returns a vector of
    predictions.
    """
    # check classes
    if sum(~np.isin(y, [-1, 1])) != 0:
        raise Exception(
            'The classes are not encoded as (-1,1) or there are '
            'more than two classes!')

    # initialize weights
    n_obs = len(y)
    obs_weights = np.array([1 / n_obs for i in
        range(n_obs)])  # start with equal weights

    # run loop
    y_pred_M = np.zeros((M, n_obs))
    alpha_M = np.zeros(M)
    for m in range(M):
        clf.fit(X, y,
            sample_weight=obs_weights)  # fit classifier using
        # sample_weights
        y_pred_m = clf.predict(X)  # obtain predictions

        err_ind = np.array(y != y_pred_m).astype(
            'int')  # vector indicating prediction errors
        err_m = obs_weights.T @ err_ind / np.sum(obs_weights)
        alpha_m = np.log((1 - err_m) / err_m)

        obs_weights = [obs_weights[i] * np.exp(alpha_m * err_ind[i])
            for i in range(len(y))]
        obs_weights = obs_weights / np.sum(
            obs_weights)  # normalize weights to sum to 1

        # save predictions and alpha
        y_pred_M[m] = y_pred_m
        alpha_M[m] = alpha_m

    # Make vector of final predictions
    y_pred = [np.sign(
        np.sum([alpha_M[m] * y_pred_M[m, i] for m in range(M)])) for
        i in range(n_obs)]

    return y_pred


#####################################################################
# Verify that our own implementation gives the same results as the
# sklearn package

# try different values for tree depth
for tree_depth in (1, 2, 3):
    # try different values for number of trees / iterations
    for M in (100, 200, 500):
        # Set classifier to a decision tree with maximum depth
        # tree_depth (and fixed seed to select the
        # same tree structure even if two splits would give the
        # same improvement)
        tree = sklearn.tree.DecisionTreeClassifier(
            max_depth=tree_depth, random_state=seed)

        # get predictions from own implementation
        y_pred = adaboost_m1(X, y, tree, M=M)

        # get predictions from sklearn package
        adaboost_m1_package = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=tree, n_estimators=M, learning_rate=1.0,
            random_state=seed,
            algorithm='SAMME')  # only SAMME gives same results
        adaboost_m1_package.fit(X, y)
        y_pred_package = adaboost_m1_package.predict(X)

        # verify that the predictions are identical
        print(
            'The AdaBoost predictions from skicit-learn and our '
            'own implementation are the same: ',
            np.sum(y_pred != y_pred_package) == 0,
            '(Tree depth = {}, Iterations = {})'.format(tree_depth,
                M))


#####################################################################
# Plotting/printing functions

def plot_learning(estimator, X, y, save_as=None):
    """ Plots learning curve and scalability of a given estimator.
    """
    # Construct learning curve
    train_sizes = np.linspace(0.01, 1, 10)
    learning_curve = sklearn.model_selection.learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=5,
        scoring='balanced_accuracy',
        exploit_incremental_learning=False, n_jobs=-1, verbose=1,
        random_state=seed, return_times=True)
    train_sizes, train_scores, test_scores, fit_times, \
        _ = learning_curve
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
    axes[0].fill_between(train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes[0].fill_between(train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
        label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
        label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[0].set_title("Learning curve")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit time in seconds")
    axes[1].set_title("Scalability of the model")

    plt.tight_layout()
    if save_as != None:
        plt.savefig(save_as, dpi=150)
    plt.show()

    return


def print_cv(cv, X, y, model_name):
    """ Prints best score, best parameter values, and in and out
    of sample confusion matrices for a cv
    result """
    print('\n Results for {}:'.format(model_name))
    print('The best out-of-sample performance is {}'.format(
        cv.best_score_))
    print('Best parameter values: ', cv.best_params_)

    pred = cv.best_estimator_.predict(X)
    print('In-sample confusion matrix of best estimator: ',
        sklearn.metrics.confusion_matrix(y, pred))

    cross_val_pred = sklearn.model_selection.cross_val_predict(
        cv.best_estimator_, X, y, cv=5, n_jobs=8)
    print(
        'Out-of-sample confusion matrix of best estimator:\n{'
        '}'.format(
            sklearn.metrics.confusion_matrix(y, cross_val_pred)))

    return


def plot_permutation_importance(estimator, X, y, save_as=None):
    """ Boxplot of feature permutation importance for a given
    estimator. """
    result = permutation_importance(estimator, X, y,
        scoring='balanced_accuracy', n_repeats=100, n_jobs=10,
        random_state=seed)
    perm_sorted_idx = result.importances_mean.argsort()

    plt.figure(figsize=(10, 8))
    plt.boxplot(result.importances[perm_sorted_idx].T, vert=False,
        labels=df.columns[perm_sorted_idx])
    fig.tight_layout()
    if save_as != None:
        plt.savefig(save_as, dpi=150)
    plt.xlabel('Feature permutation importance')
    plt.show()

    return


#####################################################################
# Feature engineering

# Z-score based outlier removal of feature values
def zscore_outlier_removal(X, threshold=5):
    """ Sets feature values in X that are more than threshold
    times standard deviation away from their mean
    to NaN. Returns X with original length but some column values
    are NaN.
    """
    new_X = copy.deepcopy(X)
    new_X[abs(sklearn.preprocessing.scale(X)) > threshold] = np.nan

    return new_X


# Make zscore feature outlier removal a transformer function
zscore_outlier_removal = sklearn.preprocessing.FunctionTransformer(
    zscore_outlier_removal, kw_args=dict(threshold=5))

# Replace feature outliers with imputed values via KNN
KNN_impute = KNNImputer()

# Polynomial feature expansion
poly = sklearn.preprocessing.PolynomialFeatures(degree=2,
    interaction_only=False, include_bias=False)

# demean and scale to unit variance
scale = sklearn.preprocessing.StandardScaler()

#####################################################################
# Support Vector Machine
svm = sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',
    class_weight='balanced', decision_function_shape='ovr')

svm_pipe = sklearn.pipeline.Pipeline(
    [('outlier', zscore_outlier_removal), ('impute', KNN_impute),
        ('scale', scale), ('svm', svm)])

# values to try for cross-validation
penalty_vals = [np.e ** i for i in np.linspace(-3, 3, 16)]
poly_degrees = [1, 2, 3]
zscore_threshold_vals = [100, 10, 9, 8, 7, 6,
    5]  # 100 = no outlier treatment
kernels = ['rbf', 'poly', 'sigmoid']

# full parameter grid
svm_grid = {"outlier__kw_args": [dict(threshold=i) for i in
    zscore_threshold_vals], "svm__C": penalty_vals,
    "svm__kernel": kernels, "svm__degree": poly_degrees}

# optimal values (to save time)
svm_grid = {"outlier__kw_args": [dict(threshold=i) for i in [7]],
    "svm__C": penalty_vals, "svm__kernel": ['rbf']}

# stratified 5-fold cross-validation
svm_cv = sklearn.model_selection.GridSearchCV(svm_pipe, svm_grid,
    scoring='balanced_accuracy', n_jobs=8, refit=True, verbose=True,
    return_train_score=True, cv=5)
svm_cv.fit(X, y)

# Results
print_cv(svm_cv, X, y, 'support vector machine')

plot_learning(svm_cv.best_estimator_, X, y, save_as='svm_learning')

plot_permutation_importance(svm_cv.best_estimator_, X, y,
    save_as='svm_importance')

#####################################################################
# AdaBoost

tree = sklearn.tree.DecisionTreeClassifier(max_depth=2,
    class_weight='balanced', random_state=seed)

boost = sklearn.ensemble.AdaBoostClassifier(base_estimator=tree,
    n_estimators=1000, learning_rate=1.0, random_state=seed,
    algorithm='SAMME.R')

# Put all processing and feature engineering in a pipeline
boost_pipe = imblearn.pipeline.Pipeline(
    [('outlier', zscore_outlier_removal), ('impute', KNN_impute),
        ('poly', poly), ('boost', boost)])

len(df[df.Status == -1]) / len(df)

# values to try for cross-validation
learning_rates = [0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 1]
n_iterations_vals = [200, 500, 1000, 1500, 2000]
tree_depths = [1, 2, 3, 4, 5]
poly_degrees = [1, 2, 3]
zscore_threshold_vals = [100, 7, 5]

# full parameter grid (takes too long, run one at a time instead)
boost_grid = {"outlier__kw_args": [dict(threshold=i) for i in
    zscore_threshold_vals], "poly__degree": poly_degrees,
    "boost__base_estimator__max_depth": tree_depths,
    "boost__learning_rate": learning_rates,
    "boost__n_estimators": n_iterations_vals}

# optimal values found through one at a time CV
boost_grid = {"outlier__kw_args": [dict(threshold=i) for i in [100]],
    "poly__degree": [2], "boost__base_estimator__max_depth": [2],
    "boost__learning_rate": [0.7], "boost__n_estimators": [1500]}

# stratified 5-fold cross-validation
cv_boost = sklearn.model_selection.GridSearchCV(boost_pipe,
    boost_grid, scoring='balanced_accuracy', n_jobs=8, refit=True,
    verbose=True, return_train_score=False, cv=5)
cv_boost.fit(X, y)

# Results
print_cv(cv_boost, X, y, 'AdaBoost')

plot_learning(cv_boost.best_estimator_, X, y,
    save_as='boost_learning')

plot_permutation_importance(cv_boost.best_estimator_, X, y,
    save_as='boost_importance')
