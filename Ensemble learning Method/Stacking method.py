import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.model_selection import KFold

import lightgbm  as lgb
from sklearn.ensemble import RandomForestClassifier
import catboost as cb
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.svm import SVC


X=np.load('X_pca_gender.npy')
y=np.load('y_pca_gender.npy')
class_names=list(['serious','smiling'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1666,random_state=0)

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
clf3 = lgb.LGBMClassifier(objective = 'binary',
                         is_unbalance = True,
                         metric = 'binary_logloss,auc',
                         max_depth = 8,
                         num_leaves = 30,
                         feature_fraction = 0.8,
                         min_child_samples=21,
                         min_child_weight=0.001,
                         bagging_fraction = 0.9,
                         bagging_freq = 4,
                         reg_alpha = 0.001,
                         reg_lambda = 8,
                         cat_smooth = 0,
                             num_iterations=298,
                          learning_rate=0.052727272,
                        )
clf4=cb.CatBoostClassifier(verbose = False,loss_function="CrossEntropy",
                           eval_metric="Precision",
                           learning_rate=0.06939393,
                           iterations=777,
                           random_seed=42,
                           l2_leaf_reg=1,
                           od_type="Iter",
                           leaf_estimation_method='Newton',
                           depth=8,
                           early_stopping_rounds=500)
meta_classifier = SVC(probability=True, kernel='linear')
sclf = StackingClassifier(classifiers=[clf1, clf2,clf3,clf4],  meta_classifier=meta_classifier)

for clf, label in zip([clf1, clf2, clf3, clf4, sclf], ['RandomForest', 'XGBoost', 'LightGBM','CatBoost','StackingClassifier']):
       scores = cross_val_score(sclf, X, y, cv=10)
       print("score: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


