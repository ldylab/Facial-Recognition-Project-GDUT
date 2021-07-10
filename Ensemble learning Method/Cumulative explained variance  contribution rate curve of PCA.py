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
from sklearn.model_selection import RandomizedSearchCV
all_data_set = []  # 由每一张图片的list所构成
all_data_label = []  # 数据标签（smiling：标记为1，serious：标记为0）
Face_file = u"./ClassifyFiles/CookFaceAllWithoutMissingWithSame_right.csv"
Face_df = pd.read_csv(Face_file, sep=',')
#sep: 指定分割符，默认是’,’C引擎不能自动检测分隔符，但Python解析引擎可以
from_path = u"./rawdata"
for elem in range(len(Face_df)):
    from_file_name = from_path + '/' + str(Face_df["PicNum"][elem])
    img_raw = np.fromfile(from_file_name, dtype=np.uint8) # 读取图片
    #一张图片的数组img_raw

    all_data_set.append(list(img_raw.tolist()))# 数组——>列表
    #一张图片的list img_raw.tolist()
    #all_data_set每一个元素都是一张图片的list

    Face = Face_df["Face"][elem]
    if Face == 'smiling':
        all_data_label.append(1)
    elif Face == 'serious':
        all_data_label.append(0)


import seaborn as sns
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
sns.set_style('white')
sns.set_context("paper")
fig=plt.figure(figsize=(8,5),dpi=400)
np.set_printoptions(suppress=True)

n_components=140
pca_line = PCA(n_components).fit(all_data_set)
plt.grid()
plt.plot(list(range(1,n_components+1,1)),np.cumsum(pca_line.explained_variance_ratio_),color='#238E68',marker ='o',markersize=5,alpha = 0.7,linewidth=2)
plt.xlabel("Number of components after dimension reduction")
plt.ylabel("Cumulative explained variance ratio")
plt.show()