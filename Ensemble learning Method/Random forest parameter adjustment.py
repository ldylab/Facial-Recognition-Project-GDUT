import pandas as pd
import numpy as np
from time import time
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import NMF
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

#Random forest tuning (random search)
X=np.load('X_NMF.npy')
y=np.load('y_NMF.npy')
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
n_iter=10000 #Number of random searches
best_score_list=[]
best_param_list=[]
for i in range(1,n_iter+1):
    rfc = RandomForestClassifier(n_estimators=260,random_state=90,n_jobs=-1)
    param_dist ={"max_depth": sp_randint(1, 40),
    "bootstrap": [True, False],
    "criterion":['gini','entropy'],
    "max_features":["sqrt","log2"],
    "min_samples_split":sp_randint(2, 11),
    "min_samples_leaf":sp_randint(1, 7)}

    RS = RandomizedSearchCV(rfc,param_distributions=param_dist,cv = 10,n_iter=1,n_jobs = -1)
    RS.fit(X, y)
    best_param = RS.best_params_
    best_score = RS.best_score_
    best_score_list.append(best_score)
    best_param_list.append(best_param)
    print("Number of searches:",i,best_param, best_score)
    print("Optimal parameters and accuracy:",best_param_list[best_score_list.index(max(best_score_list))],max(best_score_list))