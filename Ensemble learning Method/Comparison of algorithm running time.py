import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime

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

# Import some data to play with
X = np.load('X_NMF_gender.npy')
y = np.load('y_NMF_gender.npy')

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
    starttime = datetime.datetime.now()
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    endtime = datetime.datetime.now()
    print(label, endtime - starttime)

# Import some data to play with
X = np.load('X_lda_gender.npy')
y = np.load('y_lda_gender.npy')

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
    starttime = datetime.datetime.now()
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    endtime = datetime.datetime.now()
    print(label, endtime - starttime)

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
    starttime = datetime.datetime.now()
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    endtime = datetime.datetime.now()
    print(label, endtime - starttime)

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
from sklearn.linear_model import LogisticRegression

meta_classifier = LogisticRegression(random_state=0)

sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=meta_classifier)

fpr_ = []
tpr_ = []
roc_auc_ = []
for clf, label in zip([clf1, clf2, clf3, clf4, sclf],
                      ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'StackingClassifier']):
    starttime = datetime.datetime.now()
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    endtime = datetime.datetime.now()
    print(label, endtime - starttime)

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

meta_classifier = LogisticRegression(random_state=0)

sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=meta_classifier)

fpr_ = []
tpr_ = []
roc_auc_ = []
for clf, label in zip([clf1, clf2, clf3, clf4, sclf],
                      ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'StackingClassifier']):
    starttime = datetime.datetime.now()
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    endtime = datetime.datetime.now()
    print(label, endtime - starttime)


#降维算法运行时间对比(目标特征数量90)
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(8,5),dpi=300)
name_list = ['PCA','NMF','LDA']
num_list = [20,55,100]
plt.barh(range(len(num_list)), num_list,tick_label = name_list,color=['#BDECF8','#55BFCA','#B3E58C'])
plt.text(10,0-0.1,6, ha='center',va='bottom',fontsize=20)
plt.text(27.5,1-0.1,183, ha='center',va='bottom',fontsize=20)
plt.text(50,2-0.1,2867, ha='center',va='bottom',fontsize=20)
plt.title('Running time/second')
plt.xticks([])
plt.show()