# 提取表情数据集及标签
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


# PCA降维
starttime = datetime.datetime.now()
n_components = 90
pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(all_data_set)
# PCA降维后的总数据集
all_data_pca = pca.transform(all_data_set)
eigenfaces = pca.components_.reshape((n_components, 128, 128))
# X为降维后的数据，y是对应类标签
X_pca = np.array(all_data_pca)
y_pca = np.array(all_data_label)
np.random.seed(120)
np.random.shuffle(X_pca)#shuffle() 方法将序列的所有元素随机排序
np.random.seed(120)
np.random.shuffle(y_pca)

endtime = datetime.datetime.now()
print (endtime - starttime)
np.save('X_pca.npy',X_pca)
np.save('y_pca.npy',y_pca)

# LDA降维
from sklearn.decomposition import LatentDirichletAllocation
starttime = datetime.datetime.now()
lda = LatentDirichletAllocation(n_components=90, random_state=0,n_jobs=-1)
all_data_lda = lda.fit_transform(all_data_set)
X_lda = np.array(all_data_lda)
y_lda = np.array(all_data_label)
np.random.seed(120)
np.random.shuffle(X_lda)#shuffle() 方法将序列的所有元素随机排序
np.random.seed(120)
np.random.shuffle(y_lda)

endtime = datetime.datetime.now()
print (endtime - starttime)
np.save('X_lda.npy',X_lda)
np.save('y_lda.npy',y_lda)

#NMF 降维
from sklearn.decomposition import NMF
starttime = datetime.datetime.now()
nmf=NMF(n_components=90, init='random', random_state=0)
all_data_nmf = nmf.fit_transform(all_data_set)
# X为降维后的数据，y是对应类标签
X_NMF = np.array(all_data_nmf)
y_NMF = np.array(all_data_label)
np.random.seed(120)
np.random.shuffle(X_NMF)#shuffle() 方法将序列的所有元素随机排序
np.random.seed(120)
np.random.shuffle(y_NMF)

endtime = datetime.datetime.now()
print (endtime - starttime)
np.save('X_NMF.npy',X_NMF)
np.save('y_NMF.npy',y_NMF)