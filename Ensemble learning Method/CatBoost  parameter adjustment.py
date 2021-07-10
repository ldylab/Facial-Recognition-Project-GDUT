import pandas as pd
import numpy as np
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import lightgbm  as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import catboost as cb

# catboost：kaggle参数设置参考: https://zhuanlan.zhihu.com/p/136697031
X=np.load('X_NMF.npy')
y=np.load('y_NMF.npy')
cb = cb.CatBoostClassifier(verbose = False,loss_function="Logloss",
                           eval_metric="AUC",
                           task_type="GPU",
                           learning_rate=0.01,
                           iterations=300,
                           random_seed=42,
                           od_type="Iter",
                           depth=10,
                           early_stopping_rounds=500)
score_mean = cross_val_score(cb, X, y, cv=10).mean()
print(score_mean)