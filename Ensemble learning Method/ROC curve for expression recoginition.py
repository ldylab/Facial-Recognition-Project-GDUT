import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn import metrics

import lightgbm  as lgb
from sklearn.ensemble import RandomForestClassifier
import catboost as cb
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import some data to play with
X = np.load('X_pca.npy')
y = np.load('y_pca.npy')

# Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1666, random_state=0)

# Learn to predict each class against the other
# svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)

clf1 = RandomForestClassifier(n_estimators=258,
                              bootstrap=False,
                              criterion='entropy',
                              max_depth=16,
                              max_features='sqrt',
                              min_samples_leaf=4,
                              min_samples_split=3)
clf2 = xgb.XGBClassifier(random_state=90,
                         n_estimators=301,
                         learning_rate=0.2077,
                         bootstrap=True,
                         criterion='gini',
                         max_depth=33,
                         max_features='sqrt',
                         min_samples_leaf=6,
                         min_samples_split=8)
clf3 = lgb.LGBMClassifier(objective='binary',
                          is_unbalance=True,
                          metric='binary_logloss,auc',
                          max_depth=8,
                          num_leaves=30,
                          feature_fraction=0.8,
                          min_child_samples=21,
                          min_child_weight=0.001,
                          bagging_fraction=0.9,
                          bagging_freq=4,
                          reg_alpha=0.001,
                          reg_lambda=8,
                          cat_smooth=0,
                          num_iterations=298,
                          learning_rate=0.052727272,
                          )
clf4 = cb.CatBoostClassifier(verbose=False, loss_function="CrossEntropy",
                             eval_metric="Precision",
                             learning_rate=0.06939393,
                             iterations=777,
                             random_seed=42,
                             l2_leaf_reg=1,
                             od_type="Iter",
                             leaf_estimation_method='Newton',
                             depth=8,
                             early_stopping_rounds=500)
# meta_classifier = DecisionTreeClassifier(random_state=0)
# meta_classifier = SVC(gamma='auto',probability=True)

meta_classifier = LogisticRegression(random_state=0)

sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=meta_classifier)

fpr_ = []
tpr_ = []
roc_auc_ = []
for clf, label in zip([clf1, clf2, clf3, clf4, sclf],
                      ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'StackingClassifier']):
    # ROC
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score[:, 1])
    fpr_.append(fpr)
    tpr_.append(tpr)
    roc_auc = metrics.auc(fpr, tpr)
    roc_auc_.append(roc_auc)
import seaborn as sns

sns.set_style('white')
sns.set_context("paper")
plt.figure(figsize=(8, 5), dpi=300)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
label = ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'StackingClassifier']
for i in range(len(fpr_)):
    plt.plot(fpr_[i], tpr_[i], color=colors[i], lw=1.5,
             label='ROC curve (area = %0.3f)(PCA+{name})'.format(name=label[i]) % roc_auc_[i])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (expression recognition)')
plt.legend(loc="lower right")
plt.show()

# Import some data to play with
X = np.load('X_lda.npy')
y = np.load('y_lda.npy')

# Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1666, random_state=0)

# Learn to predict each class against the other
# svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)

clf1 = RandomForestClassifier(n_estimators=258,
                              bootstrap=False,
                              criterion='entropy',
                              max_depth=16,
                              max_features='sqrt',
                              min_samples_leaf=4,
                              min_samples_split=3)
clf2 = xgb.XGBClassifier(random_state=90,
                         n_estimators=301,
                         learning_rate=0.2077,
                         bootstrap=True,
                         criterion='gini',
                         max_depth=33,
                         max_features='sqrt',
                         min_samples_leaf=6,
                         min_samples_split=8)
clf3 = lgb.LGBMClassifier(objective='binary',
                          is_unbalance=True,
                          metric='binary_logloss,auc',
                          max_depth=8,
                          num_leaves=30,
                          feature_fraction=0.8,
                          min_child_samples=21,
                          min_child_weight=0.001,
                          bagging_fraction=0.9,
                          bagging_freq=4,
                          reg_alpha=0.001,
                          reg_lambda=8,
                          cat_smooth=0,
                          num_iterations=298,
                          learning_rate=0.052727272,
                          )
clf4 = cb.CatBoostClassifier(verbose=False, loss_function="CrossEntropy",
                             eval_metric="Precision",
                             learning_rate=0.06939393,
                             iterations=777,
                             random_seed=42,
                             l2_leaf_reg=1,
                             od_type="Iter",
                             leaf_estimation_method='Newton',
                             depth=8,
                             early_stopping_rounds=500)
# meta_classifier = DecisionTreeClassifier(random_state=0)
# meta_classifier = SVC(gamma='auto',probability=True)
from sklearn.linear_model import LogisticRegression

meta_classifier = LogisticRegression(random_state=0)

sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=meta_classifier)

fpr_ = []
tpr_ = []
roc_auc_ = []
for clf, label in zip([clf1, clf2, clf3, clf4, sclf],
                      ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'StackingClassifier']):
    # ROC
    if label == "StackingClassifier":
        y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    else:
        y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score[:, 1])
    fpr_.append(fpr)
    tpr_.append(tpr)
    roc_auc = metrics.auc(fpr, tpr)
    roc_auc_.append(roc_auc)
import seaborn as sns

sns.set_style('white')
sns.set_context("paper")
plt.figure(figsize=(8, 5), dpi=300)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
label = ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'StackingClassifier']
for i in range(len(fpr_)):
    plt.plot(fpr_[i], tpr_[i], color=colors[i], lw=1.5,
             label='ROC curve (area = %0.3f)(LDA+{name})'.format(name=label[i]) % roc_auc_[i])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (expression recognition)')
plt.legend(loc="lower right")
plt.show()

# Import some data to play with
X = np.load('X_NMF.npy')
y = np.load('y_NMF.npy')

# Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1666, random_state=0)

# Learn to predict each class against the other
# svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)

clf1 = RandomForestClassifier(n_estimators=258,
                              bootstrap=False,
                              criterion='entropy',
                              max_depth=16,
                              max_features='sqrt',
                              min_samples_leaf=4,
                              min_samples_split=3)
clf2 = xgb.XGBClassifier(random_state=90,
                         n_estimators=301,
                         learning_rate=0.2077,
                         bootstrap=True,
                         criterion='gini',
                         max_depth=33,
                         max_features='sqrt',
                         min_samples_leaf=6,
                         min_samples_split=8)
clf3 = lgb.LGBMClassifier(objective='binary',
                          is_unbalance=True,
                          metric='binary_logloss,auc',
                          max_depth=8,
                          num_leaves=30,
                          feature_fraction=0.8,
                          min_child_samples=21,
                          min_child_weight=0.001,
                          bagging_fraction=0.9,
                          bagging_freq=4,
                          reg_alpha=0.001,
                          reg_lambda=8,
                          cat_smooth=0,
                          num_iterations=298,
                          learning_rate=0.052727272,
                          )
clf4 = cb.CatBoostClassifier(verbose=False, loss_function="CrossEntropy",
                             eval_metric="Precision",
                             learning_rate=0.06939393,
                             iterations=777,
                             random_seed=42,
                             l2_leaf_reg=1,
                             od_type="Iter",
                             leaf_estimation_method='Newton',
                             depth=8,
                             early_stopping_rounds=500)
# meta_classifier = DecisionTreeClassifier(random_state=0)
# meta_classifier = SVC(gamma='auto',probability=True)
from sklearn.linear_model import LogisticRegression

meta_classifier = SVC(probability=True, kernel='linear')

sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=meta_classifier)

fpr_ = []
tpr_ = []
roc_auc_ = []
for clf, label in zip([clf1, clf2, clf3, clf4, sclf],
                      ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'StackingClassifier']):
    # ROC
    if label == "StackingClassifier":
        y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    else:
        y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score[:, 1])
    fpr_.append(fpr)
    tpr_.append(tpr)
    roc_auc = metrics.auc(fpr, tpr)
    roc_auc_.append(roc_auc)
