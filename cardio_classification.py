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
import imblearn
import copy
import seaborn as sns
import sklearn
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.impute import KNNImputer
from sklearn.calibration import calibration_curve

#############################################################################################################
# Data

# set number of cores to use
cores = 8

# fix seed to make results reproducible
seed = 42
# Read data
url = "https://raw.githubusercontent.com/jakob-ra/Cardiotocography/master/cardiotocography.csv"
df = pd.read_csv(url, sep=';', decimal=',')

print(df.dtypes)  # check data types

print('Number of missing values: ', df.isnull().sum().sum())  # check for missing

# use subset of columns
cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max',
    'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'NSP']
df = df[cols]

# Make binary outcome variable (normal,suspect+pathological)
df['Status'] = np.where(df.NSP == 1, -1, 1)  # recodes normal to -1 and everything else to 1

# shuffle data once to break up any inherent order the observations might have (important for k-fold
# crossvalidation)
df = df.sample(frac=1, random_state=seed)

# make vector of class labels and feature matrix
df_X = df.drop(columns=['NSP', 'Status'])
y, X = df.Status.values, df_X.values.astype('float')

#############################################################################################################
# Descriptive plots

# Histogram for all variables
fig, ax = plt.subplots(1, 1, figsize=(20, 20))
df.hist(ax=ax)
plt.savefig('hist', dpi=150)
plt.show()

# Boxplots for feature distributions
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
df_scale_X = pd.DataFrame(sklearn.preprocessing.scale(df), columns=df.columns).drop(
    columns=['NSP', 'Status'])
df_scale_X.boxplot(ax=ax, rot=45)
plt.savefig('boxplot', dpi=150)
plt.show()

# Feature correlation heatmap
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
corr = df.drop(columns=['NSP']).corr()
corr = corr.round(decimals=2)
corr = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, ax=ax)
plt.xticks(rotation=45)
plt.savefig('corr', dpi=150)
plt.show()

# Plot example tree
plt.figure(figsize=(10, 15))
example_tree = sklearn.tree.DecisionTreeClassifier(max_depth=2)
example_tree.fit(X, y)
sklearn.tree.plot_tree(example_tree, feature_names=df_X.columns,
    class_names=['normal', 'suspect/pathological'], filled=True, node_ids=True, precision=2, impurity=True)
plt.tight_layout()
# plt.savefig('tree', dpi=150)
plt.show()


#############################################################################################################
# Own implementation of AdaBoost algorithm

def adaboost_m1(X, y, clf, M=500):
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

# try different values for tree depth
for tree_depth in (1, 2, 3):
    # try different values for number of trees / iterations
    for M in (100, 200, 500):
        # Set classifier to a decision tree with maximum depth tree_depth (and fixed seed to select the
        # same tree structure even if two splits would give the same improvement)
        tree = sklearn.tree.DecisionTreeClassifier(max_depth=tree_depth, random_state=seed)

        # get predictions from own implementation
        y_pred = adaboost_m1(X, y, tree, M=M)

        # get predictions from sklearn package
        adaboost_m1_package = sklearn.ensemble.AdaBoostClassifier(base_estimator=tree, n_estimators=M,
            learning_rate=1.0, random_state=seed, algorithm='SAMME')  # only SAMME gives same results
        adaboost_m1_package.fit(X, y)
        y_pred_package = adaboost_m1_package.predict(X)

        # verify that the predictions are identical
        print('The AdaBoost predictions from skicit-learn and our own implementation are the same: ',
            np.sum(y_pred != y_pred_package) == 0,
            '(Tree depth = {}, Iterations = {})'.format(tree_depth, M))


#############################################################################################################
# Plotting/printing functions

def plot_learning(estimator, X, y, save_as=None):
    """ Plots learning curve and scalability of a given estimator.
    """
    # Construct learning curve
    train_sizes = np.linspace(0.01, 1, 10)
    learning_curve = sklearn.model_selection.learning_curve(estimator, X, y, train_sizes=train_sizes, cv=5,
        scoring='balanced_accuracy', exploit_incremental_learning=False, n_jobs=cores, verbose=1,
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
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
        alpha=0.1, color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[0].set_title("Learning curve")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std,
        alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit time in seconds")
    axes[1].set_title("Scalability of the model")

    plt.tight_layout()
    if save_as != None:
        plt.savefig(save_as, dpi=150)
    plt.show()

    return


def print_cv(cv, X, y, model_name):
    """ Prints best score, best parameter values, and in and out of sample confusion matrices for a cv
    result """
    print('\n Results for {}:'.format(model_name))
    print('The best out-of-sample performance is {}'.format(cv.best_score_))
    print('Best parameter values: ', cv.best_params_)

    pred = cv.best_estimator_.predict(X)
    print('In-sample confusion matrix of best estimator: ', sklearn.metrics.confusion_matrix(y, pred))

    cross_val_pred = sklearn.model_selection.cross_val_predict(cv.best_estimator_, X, y, cv=5, n_jobs=cores)
    print('Out-of-sample confusion matrix of best estimator:\n{}'.format(
        sklearn.metrics.confusion_matrix(y, cross_val_pred)))

    return


def plot_permutation_importance(estimator, X, y, save_as=None):
    """ Boxplot of feature permutation importance for a given estimator. """
    result = permutation_importance(estimator, X, y, scoring='balanced_accuracy', n_repeats=50, n_jobs=cores,
        random_state=seed)
    perm_sorted_idx = result.importances_mean.argsort()

    plt.figure(figsize=(10, 8))
    plt.boxplot(result.importances[perm_sorted_idx].T, vert=False, labels=df.columns[perm_sorted_idx])
    fig.tight_layout()
    if save_as != None:
        plt.savefig(save_as, dpi=150)
    plt.xlabel('Feature permutation importance')
    plt.show()

    return

def plot_oos_roc(estimator, X ,y, ax=None, name=None):
    """ Plots a roc curve based on out-of-sample prediction scores of a given estimator. """
    cross_val_pred_scores = sklearn.model_selection.cross_val_predict(estimator, X, y, cv=5,
        method='predict_proba', n_jobs=cores)[:,1]
    fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(y, cross_val_pred_scores)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    viz = sklearn.metrics.RocCurveDisplay(fpr, tpr, roc_auc, estimator)

    return viz.plot(ax=ax, name=name)


#############################################################################################################
# Feature engineering

# Z-score based outlier removal of feature values
def zscore_outlier_removal(X, threshold=7):
    """ Sets feature values in X that are more than threshold times standard deviation away from their mean
    to NaN. Returns X with original length but some column values are NaN.
    """
    new_X = copy.deepcopy(X)
    new_X[abs(sklearn.preprocessing.scale(X)) > threshold] = np.nan

    return new_X

# Make zscore feature outlier removal a transformer function
zscore_outlier_removal = sklearn.preprocessing.FunctionTransformer(zscore_outlier_removal,
    kw_args=dict(threshold=7))

# Replace feature outliers with imputed values via KNN
KNN_impute = KNNImputer()

# Polynomial feature expansion
poly = sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

# demean and scale to unit variance
scale = sklearn.preprocessing.StandardScaler()

#############################################################################################################
# Support Vector Machine
svm = sklearn.svm.SVC(C=9, kernel='rbf', gamma='scale', class_weight='balanced', probability=True,
    decision_function_shape='ovr')

svm_pipe = sklearn.pipeline.Pipeline(
    [('outlier', zscore_outlier_removal), ('impute', KNN_impute), ('scale', scale), ('svm', svm)])

# values to try for cross-validation
penalty_vals = [np.e ** i for i in np.linspace(-3, 3, 16)]
poly_degrees = [1, 2, 3]
zscore_threshold_vals = [100, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # 100 = no outlier treatment
kernels = ['rbf', 'poly', 'sigmoid']

# full parameter grid
svm_grid = {"outlier__kw_args": [dict(threshold=i) for i in zscore_threshold_vals], "svm__C": penalty_vals,
    "svm__kernel": kernels, "svm__degree": poly_degrees}

# optimal values (to save time)
svm_grid = {"outlier__kw_args": [dict(threshold=i) for i in zscore_threshold_vals], "svm__C": [9],
    "svm__kernel": ['rbf']}

# stratified 5-fold cross-validation
svm_cv = sklearn.model_selection.GridSearchCV(svm_pipe, svm_grid, scoring='balanced_accuracy', n_jobs=cores,
    refit=True, verbose=True, return_train_score=True, cv=5)
svm_cv.fit(X, y)

# Results
print_cv(svm_cv, X, y, 'support vector machine')

plot_learning(svm_cv.best_estimator_, X, y, save_as='svm_learning')

plot_permutation_importance(svm_cv.best_estimator_, X, y, save_as='svm_importance')

svm_cv.cv_results_
#############################################################################################################
# SVM bagging

svm = sklearn.svm.SVC(C=9, kernel='rbf', gamma='scale', class_weight='balanced',
    decision_function_shape='ovr')

smote = imblearn.over_sampling.SMOTE(sampling_strategy=1, random_state=seed)

svm_bag_pipe = imblearn.pipeline.Pipeline([('outlier', zscore_outlier_removal),
    ('impute', KNN_impute), ('smote', smote), ('scale', scale), ('svm', svm)])

svm_bag = sklearn.ensemble.BaggingClassifier(base_estimator=svm_bag_pipe, n_estimators=10,
    max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False,
    warm_start=False, n_jobs=cores, random_state=seed, verbose=0)

svm_bag_grid = {"max_features": [1.0,.8,.5],
    "max_samples": [1.0],
    "n_estimators": [10,30,50]
    }

svm_bag_cv = sklearn.model_selection.GridSearchCV(svm_bag, svm_bag_grid, scoring='balanced_accuracy',
    n_jobs=cores, refit=True, verbose=True, return_train_score=True, cv=5)
svm_bag_cv.fit(X,y)

print_cv(svm_bag_cv, X, y, 'SVM bag')


#############################################################################################################
# Random forest

random_forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=2,
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=10,
    bootstrap=True, oob_score=False, n_jobs=cores, random_state=seed, class_weight='balanced')

tree_numbers = [20, 50, 100]
tree_depths = [2, 3, 5, 7]
max_feature_vals = [.2,.5,.7]

forest_pipe = sklearn.pipeline.Pipeline(
    [('poly', poly), ('forest', random_forest)])

random_forest_grid = {"poly__degree": [1,2],
    "forest__n_estimators": tree_numbers,
    "forest__max_depth": tree_depths,
    "forest__max_features": max_feature_vals}

random_forest_cv = sklearn.model_selection.GridSearchCV(forest_pipe, random_forest_grid,
    scoring='balanced_accuracy', n_jobs=cores, refit=True, verbose=True, return_train_score=True, cv=5)
random_forest_cv.fit(X,y)

print_cv(random_forest_cv, X, y, 'Random forest')

plot_learning(random_forest_cv.best_estimator_, X, y, save_as='svm_learning')

plot_permutation_importance(random_forest_cv.best_estimator_, X, y, save_as='svm_importance')

#############################################################################################################
# LightGB

gbm = lgb.LGBMClassifier(max_depth=5, class_weight='balanced',
                        learning_rate=1, n_estimators=500, random_state=seed, num_leaves=100)

# values to try for cross-validation
learning_rates = [0.011]
n_iterations_vals = [500]
tree_depths = [9]

gbm_grid = {"max_depth": tree_depths,
            "learning_rate": learning_rates,
            "n_estimators": n_iterations_vals}

# stratified 5-fold cross-validation
cv_boost = sklearn.model_selection.GridSearchCV(gbm, gbm_grid, scoring='balanced_accuracy',
    n_jobs=cores, refit=True, verbose=True, return_train_score=False, cv=5)
cv_boost.fit(X,y)

print_cv(cv_boost, X, y, 'AdaBoost')

plot_learning(cv_boost.best_estimator_, X, y, save_as='AdaBoost')

plot_permutation_importance(cv_boost.best_estimator_, X, y, save_as='AdaBoost')

#############################################################################################################
# Logistic regression

log_reg = sklearn.linear_model.LogisticRegression(fit_intercept=True, dual=False, C=0.3, l1_ratio=0.9,
    penalty='elasticnet', solver='saga', tol=0.001, max_iter=5000, class_weight='balanced')

# optimal pipeline without outlier treatment
log = imblearn.pipeline.Pipeline([('poly', poly), ('scale', scale), ('log_reg', log_reg)])

# this is the optimal parameter grid
log_grid = {"poly__degree": [2], "log_reg__C": [3.4], "log_reg__l1_ratio": [0.9]}

# stratified 5-fold cross-validation
cv_log = sklearn.model_selection.GridSearchCV(log, log_grid, scoring='balanced_accuracy', n_jobs=cores,
    refit=True, verbose=1, return_train_score=True, cv=5)
cv_log.fit(X, y)

print_cv(cv_log, X, y, 'elastic net penalized logistic regression')

plot_learning(cv_log.best_estimator_, X, y, save_as=None)

plot_permutation_importance(cv_log.best_estimator_, X, y)

#############################################################################################################
# stacking classifier
stack_estimators = [('forest', random_forest_cv.best_estimator_), ('svm', svm_cv.best_estimator_),
    ('Logit', cv_log.best_estimator_), ('svm_bag', svm_bag_cv.best_estimator_)]
stack = sklearn.ensemble.StackingClassifier(stack_estimators, cv=5, stack_method='auto', n_jobs=cores, verbose=1)
stack.fit(X,y)

models = [('forest', random_forest_cv.best_estimator_), ('svm', svm_cv.best_estimator_),
    ('boost', cv_boost.best_estimator_)]
voting = sklearn.ensemble.VotingClassifier(models, voting='hard')

cross_val_pred = sklearn.model_selection.cross_val_predict(voting, X, y, cv=5, n_jobs=cores)
print('Out-of-sample confusion matrix of best estimator:\n{}'.format(
    sklearn.metrics.confusion_matrix(y, cross_val_pred)))

sklearn.metrics.balanced_accuracy_score(y,cross_val_pred)

cross_val_pred_prob = sklearn.model_selection.cross_val_predict(cv_boost.best_estimator_, X, y, cv=5,
    n_jobs=cores, method='predict_proba')[:,1]
fraction_of_positives, mean_predicted_value = calibration_curve(y, cross_val_pred_prob, normalize=False,
    n_bins=7, strategy='uniform')
name='Boost'
clf_score = sklearn.metrics.brier_score_loss(y, cross_val_pred_prob, pos_label=y.max())
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (name, clf_score))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.legend()
plt.show()
#############################################################################################################
# Comparison

# Out-of-sample ROC curve comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_oos_roc(random_forest_cv.best_estimator_, X, y, ax=ax, name='Random Forest')
plot_oos_roc(svm_cv.best_estimator_, X, y, ax=ax, name='SVM')
plot_oos_roc(cv_boost.best_estimator_, X, y, ax=ax, name='AdaBoost')
plot_oos_roc(cv_log.best_estimator_, X, y, ax=ax, name='Logit')
plot_oos_roc(svm_bag_cv.best_estimator_, X, y, ax=ax, name='Bagged SVM')
plot_oos_roc(stack, X, y, ax=ax, name='Stack')
plt.show()

# ROC curve in-sample
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
sklearn.metrics.plot_roc_curve(svm_cv.best_estimator_, X, y, name='SVM', ax=ax)
# sklearn.metrics.plot_roc_curve(cv_log.best_estimator_, X, y, name='Log', ax=ax)
sklearn.metrics.plot_roc_curve(random_forest_cv.best_estimator_, X, y, name='Random Forest', ax=ax)
sklearn.metrics.plot_roc_curve(cv_boost.best_estimator_, X, y, name='AdaBoost', ax=ax)
plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), c='black', linestyle='--')
plt.show()




#############################################################################################################
# Unused


#############################################################################################################
# AdaBoost

tree = sklearn.tree.DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=seed)

boost = sklearn.ensemble.AdaBoostClassifier(base_estimator=tree, n_estimators=1000, learning_rate=1.0,
    random_state=seed, algorithm='SAMME.R')

# Put all processing and feature engineering in a pipeline
boost_pipe = sklearn.pipeline.Pipeline([('poly', poly), ('boost', boost)])

# values to try for cross-validation
learning_rates = [0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 1]
n_iterations_vals = [500, 1000, 1500, 2000]
tree_depths = [1, 2, 3, 4, 5, 6]
poly_degrees = [1, 2]

# full parameter grid (takes too long, run one at a time instead)
boost_grid = {"poly__degree": poly_degrees, "boost__base_estimator__max_depth": tree_depths,
    "boost__learning_rate": learning_rates, "boost__n_estimators": [1000]}

# optimal values found through one at a time CV
boost_grid = {"poly__degree": [2], "boost__base_estimator__max_depth": [1],
    "boost__learning_rate": [0.1], "boost__n_estimators": [1000]}

# optimal values found through one at a time CV
# boost_grid = {"outlier__kw_args": [dict(threshold=i) for i in [100]], "poly__degree": [2],
#     "boost__base_estimator__max_depth": [2], "boost__learning_rate": [0.7], "boost__n_estimators": [1500]}

# stratified 5-fold cross-validation
cv_boost = sklearn.model_selection.GridSearchCV(boost_pipe, boost_grid, scoring='balanced_accuracy',
    n_jobs=cores, refit=True, verbose=True, return_train_score=False, cv=5)
cv_boost.fit(X, y)

# Results
print_cv(cv_boost, X, y, 'AdaBoost')

# plot_learning(cv_boost.best_estimator_, X, y, save_as='boost_learning')

# plot_permutation_importance(cv_boost.best_estimator_, X, y, save_as='boost_importance')


# LOF observation outlier removal (not used in the end)
def lof_outlier_removal(X, y, share_out=10 ** (-2)):
    """ Removes outlier observations from X using local outlier factors. Number of removed outliers based on
    share_out (prior on the share of outliers). Returns shortened sample X,y.
    """
    if share_out == 0:
        return X, y

    lof = sklearn.neighbors.LocalOutlierFactor(n_neighbors=50, algorithm='auto', leaf_size=30,
        metric='minkowski', p=2, metric_params=None, contamination=share_out, novelty=False)

    inlier = lof.fit_predict(X)

    return X[inlier == 1], y[inlier == 1]


# Make lof outlier_removal a transformer function
lof_outlier_removal = imblearn.FunctionSampler(func=lof_outlier_removal, kw_args=dict(share_out=10 ** (-2)))

# SMOTE oversampling (not used in the end because it performs worse than balanced class weights)
smote = imblearn.over_sampling.SMOTE(sampling_strategy=1, random_state=seed)


# Plot validation curve (penalty values vs score)
penalty_vals_fine = [np.e ** i for i in np.linspace(-6, 6, 32)]  # make finer grid of penalty values
param_grid_learning = {"log_reg__C": penalty_vals_fine}
cv_log = sklearn.model_selection.GridSearchCV(log, param_grid_learning, scoring='balanced_accuracy',
    n_jobs=cores, refit=True, verbose=True, return_train_score=True, cv=5)
cv_log.fit(X, y)  # rerun the cross-validation
cv_log_results = pd.DataFrame(cv_log.cv_results_)
plt.figure(figsize=(10, 8))
plt.plot(np.log(penalty_vals_fine), cv_log_results.mean_test_score, color="black", label='Test')
plt.fill_between(np.log(penalty_vals_fine), cv_log_results.mean_test_score + cv_results.std_test_score,
    cv_log_results.mean_test_score - cv_log_results.std_test_score, color="silver")

plt.plot(np.log(penalty_vals_fine), cv_log_results.mean_train_score, color="blue", label='Train')
plt.fill_between(np.log(penalty_vals_fine), cv_log_results.mean_train_score + cv_results.std_train_score,
    cv_log_results.mean_train_score - cv_log_results.std_train_score, color="bisque")
plt.axvline(np.log(cv_log.best_params_['log_reg__C']), c='r', linestyle='--')
plt.ylabel('Balanced accuracy')
plt.xlabel('Log of inverse penalty strength')
# plt.title('Penalization vs performance for L1-ratio={}'.format(0.9))
plt.legend()
# plt.savefig('log_penalty', dpi=150)
plt.show()


# zscore feature value outlier removal and subsequent imputation
log_z = sklearn.pipeline.Pipeline(
    [('outlier', zscore_outlier_removal), ('impute', KNN_impute), ('poly', poly), ('scale', scale),
        ('log_reg', log_reg)])

# lof observation outlier removal
log_lof = imblearn.pipeline.Pipeline(
    [('outlier', lof_outlier_removal), ('poly', poly), ('scale', scale), ('log_reg', log_reg)])

# SMOTE oversampling pipeline (then have to set class weights in classifier to None rather than balanced)
log_smote = imblearn.pipeline.Pipeline(
    [('smote', smote), ('poly', poly), ('scale', scale), ('log_reg', log_reg)])

# values to try for cross-validation
l1_ratio_vals = [0, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
penalty_vals = [np.e ** i for i in np.linspace(-3, 3, 16)]
class_weights = np.linspace(0, 1, 11)
# [{-1: x, 1: 1 - x} for x in class_weights]}
lof_outlier_shares = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
poly_degrees = [1, 2, 3]
zscore_threshold_vals = [100, 10, 9, 8, 7, 6, 5]  # 100 = no outlier treatment
smote_minority_ratios = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1]

# construct parameter grids for cross-validation
param_grid_lof = {"outlier__kw_args": [dict(share_out=i) for i in [0]], "poly__degree": [2],
    "log_reg__C": [3.4], "log_reg__l1_ratio": [0.9], "log_reg__class_weight": ['balanced']}

# this would be the full parameter grid for cross-validation (do not recommend running this because of
# long runtime), instead I tried one or two parameters at a time
param_grid_z = {"outlier__kw_args": [dict(threshold=i) for i in zscore_threshold_vals],
    "poly__degree": poly_degrees, "log_reg__C": penalty_vals, "log_reg__l1_ratio": l1_ratio_vals, }

param_grid_smote = {"smote__sampling_strategy": smote_minority_ratios}

#############################################################################################################  # Manual k-fold + oversampling

# def k_fold_strat_oversamp(X, y, k=10):  #     """ Returns k bins with training and test samples where
# each bin has the original  #     class distribution (via stratisfied sampling) and for the training sets
#     we then oversample the underrepresented class with naive random oversampling  #     to get balance.
#     """  #     k_folds = sklearn.model_selection.StratifiedKFold(n_splits=k)  #     cv_indices = []  #
#     for train_idx, test_idx, in k_folds.split(X, y):  #         X_train, y_train = X[train_idx],
#     y[train_idx]  #  #         # Random oversampling  #         ros =
#     imblearn.over_sampling.RandomOverSampler(random_state=seed)  #         ros.fit_resample(X_train,
#     y_train)  #         train_idx = ros.sample_indices_  #  #         cv_indices.append((train_idx,
#     test_idx))  #  #     #    plt.hist(y_train) # each training sample now has class balance  #  #
#     return cv_indices

# Recall score
# sklearn.metrics.recall_score(y, pred)

# Load variable names
# url = "https://raw.githubusercontent.com/jakob-ra/Cardiotocography/master/varnames.csv"
# varnames = pd.read_csv(url, sep=';')
# cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'mSTV', 'ALTV', 'mLTV', 'Width', 'Min', 'Max',
#     'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'NSP']
# varnames = varnames.set_index('var').T[cols].T
# varnames = varnames.set_index('var').T.to_dict(orient='records')
