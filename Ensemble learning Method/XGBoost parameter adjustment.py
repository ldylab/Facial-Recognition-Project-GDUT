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
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

rs = []
var = []
ge = []

X=np.load('X_NMF.npy')
y=np.load('y_NMF.npy')

axis=range(1,400,50)
for i in axis:
    rfc = xgb.XGBClassifier(n_estimators=i,
                            random_state=90,tree_method='gpu_hist')
    score_mean = cross_val_score(rfc, X, y, cv=10).mean()
    score_std = cross_val_score(rfc, X, y, cv=10).std()
    rs.append(score_mean)
    var.append(score_std)
    print("n_estimators",i,"rs:",score_mean,"var:",score_std,"ge:",(1 - score_mean)**2+score_std)
    ge.append((1 - score_mean)**2+score_std)

#打印R2最高所对应的参数取值，并打印这个参数下的方差
print(axis[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])
#打印方差最低时对应的参数取值，并打印这个参数下的R2
print(axis[var.index(min(var))],rs[var.index(min(var))],min(var))
#打印泛化误差可控部分的参数取值，并打印这个参数下的R2，方差以及泛化误差的可控部分
print(axis[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
np.save('n_estimator_rs.npy',rs)
np.save('n_estimator_var.npy',var)
np.save('n_estimator_ge.npy',ge)


#绘图
sns.set_style('white')
sns.set_context("paper")
fig=plt.figure(figsize=(8,5),dpi=300)
np.set_printoptions(suppress=True)



fig = plt.figure(1) #定义figure

ax_k = HostAxes(fig, [0, 0, 0.9, 0.9]) #用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1

#parasite addtional axes, share x
ax_p = ParasiteAxes(ax_k, sharex=ax_k)
ax_K = ParasiteAxes(ax_k, sharex=ax_k)

#append axes
ax_k.parasites.append(ax_p)
ax_k.parasites.append(ax_K)

ax_k.set_ylabel('Average accuracy')
ax_k.axis['bottom'].major_ticklabels.set_rotation(45)
ax_k.set_xlabel('n_estimator')
ax_k.axis['bottom','left'].label.set_fontsize(12) # 设置轴label的大小
ax_k.axis['bottom'].major_ticklabels.set_pad(8) #设置x轴坐标刻度与x轴的距离，坐标轴刻度旋转会使label和坐标轴重合
ax_k.axis['bottom'].label.set_pad(12) #设置x轴坐标刻度与x轴label的距离，label会和坐标轴刻度重合
ax_k.axis[:].major_ticks.set_tick_out(True) #设置坐标轴上刻度突起的短线向外还是向内

#invisible right axis of ax_k
ax_k.axis['right'].set_visible(False)
ax_k.axis['top'].set_visible(True)
ax_p.axis['right'].set_visible(True)
ax_p.axis['right'].major_ticklabels.set_visible(True)
ax_p.axis['right'].label.set_visible(True)
ax_p.axis['right'].major_ticks.set_tick_out(True)
ax_p.set_ylabel('Variance')
ax_p.axis['right'].label.set_fontsize(13)
ax_K.set_ylabel('Generalization error')

K_axisline = ax_K.get_grid_helper().new_fixed_axis

ax_K.axis['right2'] = K_axisline(loc='right', axes=ax_K, offset=(60,0))
ax_K.axis['right2'].major_ticks.set_tick_out(True)
ax_K.axis['right2'].label.set_fontsize(13)
fig.add_axes(ax_k)

curve_k1, = ax_k.plot(list(axis), rs, marker ='v',markersize=8,label="Average accuracy",alpha = 0.7,linewidth=3)
curve_p, = ax_p.plot(list(axis), var, marker ='P',markersize=8,label="Variance",alpha = 0.7,linewidth=3)
curve_K, = ax_K.plot(list(axis), ge, marker ='o',markersize=8, label="Generalization error",alpha = 0.7,linewidth=3)

ax_k.axis['bottom'].major_ticklabels.set_rotation(45)

ax_p.set_ylim(0,0.035)
ax_K.set_ylim(0,0.175)

ax_k.legend(labelspacing = 0.4, fontsize = 10)

#轴名称，刻度值的颜色

ax_p.axis['right'].label.set_color(curve_p.get_color()) # 坐标轴label的颜色
ax_K.axis['right2'].label.set_color(curve_K.get_color())


ax_p.axis['right'].major_ticks.set_color(curve_p.get_color()) # 坐标轴刻度小突起的颜色
ax_K.axis['right2'].major_ticks.set_color(curve_K.get_color())

ax_p.axis['right'].major_ticklabels.set_color(curve_p.get_color()) # 坐标轴刻度值的颜色
ax_K.axis['right2'].major_ticklabels.set_color(curve_K.get_color())

ax_p.axis['right'].line.set_color(curve_p.get_color()) # 坐标轴线的颜色
ax_K.axis['right2'].line.set_color(curve_K.get_color())
plt.show()

#XGBoost tuning (random search)
from scipy.stats import uniform
from scipy.stats import randint as sp_randint

X=np.load('X_NMF.npy')
y=np.load('y_NMF.npy')
n_iter=10000 #Number of random searches
best_score_list=[]
best_param_list=[]
for i in range(1,n_iter+1):
    rfc = xgb.XGBClassifier(random_state=90,n_estimators=301,learning_rate=0.2077)

    param_dist ={
    "max_depth": sp_randint(20, 40),
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
