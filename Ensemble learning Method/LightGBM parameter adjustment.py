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
from sklearn.model_selection import RandomizedSearchCV


#LightGBM tuning (random search)
X=np.load('X_NMF.npy')
y=np.load('y_NMF.npy')
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
n_iter=10000 #Number of random searches
best_score_list=[]
best_param_list=[]
for i in range(1,n_iter+1):
    rfc = lgb.LGBMClassifier(objective = 'binary',
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
                        )
    param_dist ={"num_iterations":range(100,350,1),
             'learning_rate':np.linspace(0.1,0.5,50)}

    RS = RandomizedSearchCV(rfc,param_distributions=param_dist,cv = 10,n_iter=1,n_jobs = -1)
    RS.fit(X, y)
    best_param = RS.best_params_
    best_score = RS.best_score_
    best_score_list.append(best_score)
    best_param_list.append(best_param)
    print("Number of searches:",i,best_param, best_score)
    print("Optimal parameters and accuracy:",best_param_list[best_score_list.index(max(best_score_list))],max(best_score_list))