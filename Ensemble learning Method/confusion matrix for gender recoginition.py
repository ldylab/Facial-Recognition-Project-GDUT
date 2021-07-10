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

# NMF降维
X = np.load('X_NMF_gender.npy')
y = np.load('y_NMF_gender.npy')
class_names = list(['serious', 'smiling'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1666, random_state=0)
np.set_printoptions(precision=2)

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
lr = DecisionTreeClassifier(random_state=0)
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=lr)

for clf, label in zip([clf1, clf2, clf3, clf4, sclf],
                      ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'StackingClassifier']):
    clf.fit(X_train, y_train)
    print(label, '\n')
    i = 0
    # 绘制没有归一化的混淆矩阵
    titles_options = [("Confusion matrix(NMF+{name})".format(name=label), None),
                      ("Normalized confusion matrix (NMF+{name})".format(name=label), 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Greens,
                                     normalize=normalize)
        #         disp.ax_.set_title(title)
        matrix = disp.confusion_matrix

        print(title)
        if i == 0:
            matrix = disp.confusion_matrix
        i += 1
        print(disp.confusion_matrix)
        import seaborn as sns

        sns.set(font_scale=1.4)  # y label 横向
        import matplotlib.pyplot as plt

        # fig=plt.figure(figsize=(8,5),dpi=400)
        '''
        https://zhuanlan.zhihu.com/p/35494575
        fmt ='.0%'#显示百分比
        fmt ='f' 显示完整数字 = fmt ='g'
        fmt ='.3'显示小数的位数 = fmt ='.3f' = fmt ='.3g'
        '''

        confusion_matrix = pd.DataFrame(data=disp.confusion_matrix, index=list(range(2)), columns=list(range(2)))

        #         f, ax = plt.subplots(figsize=(8,5),dpi=300)

        f, ax = plt.subplots(figsize=(8, 5), dpi=300)
        df_cm = pd.DataFrame(confusion_matrix)
        if i == 1:
            sns.heatmap(df_cm, annot=True, vmax=300.0, vmin=0.0, fmt='.0f', cmap='Blues', annot_kws={'size': 14})
        else:
            sns.heatmap(df_cm, annot=True, vmax=1.0, vmin=0.0, fmt='.2f', cmap='Blues', annot_kws={'size': 14})

        # label_x = ax.get_xticklabels()
        # label_y = ax.get_yticklabels()
        # plt.setp(label_x, rotation=45, horizontalalignment='right',fontsize=16)
        # plt.setp(label_y,  fontsize=16)

        scale_ls = np.array(range(2)) + 0.5
        index_ls = ['female', 'male']
        plt.xticks(scale_ls, index_ls)
        plt.yticks(scale_ls, index_ls)
        plt.xlabel('Predicted Lable', fontsize=20)
        plt.ylabel('True Lable', fontsize=20)
        plt.title(title)

        # plt.savefig('./fig_confusion_matrix_of_label_embedding_prediction.png')

        plt.show()

    # 计算
    # smiling positive
    # serious negative
    TN = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TP = matrix[1][1]
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * TP / (2 * TP + FP + FN)
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Error = (FP + FN) / (TP + FP + FN + TN)
    TNR = TN / (FP + TN)

    print('精确率(precision):', Precision, '\n', '召回率(Recall):', Recall, '\n', "F1:", F1, '\n', '准确率(Accuracy):', Accuracy,
          '\n', '错误率(Error):', Error, '\n', '特异度(TNR):', TNR)

# PCA降维
X = np.load('X_pca_gender.npy')
y = np.load('y_pca_gender.npy')
class_names = list(['serious', 'smiling'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1666, random_state=0)
# clf1 = RandomForestClassifier(n_estimators=258,
#                               bootstrap=False,
#                               criterion='entropy',
#                               max_depth=16,
#                               max_features='sqrt',
#                               min_samples_leaf=4,
#                               min_samples_split=3).fit(X_train, y_train)
np.set_printoptions(precision=2)

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
lr = DecisionTreeClassifier(random_state=0)
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=lr)

for clf, label in zip([clf1, clf2, clf3, clf4, sclf],
                      ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'StackingClassifier']):
    clf.fit(X_train, y_train)
    print(label, '\n')
    i = 0
    # 绘制没有归一化的混淆矩阵
    titles_options = [("Confusion matrix (PCA+{name})".format(name=label), None),
                      ("Normalized confusion matrix (PCA+{name})".format(name=label), 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Greens,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        if i == 0:
            matrix = disp.confusion_matrix
        i += 1
        print(disp.confusion_matrix)
        import seaborn as sns

        sns.set(font_scale=1.4)  # y label 横向
        import matplotlib.pyplot as plt

        # fig=plt.figure(figsize=(8,5),dpi=400)
        '''
        https://zhuanlan.zhihu.com/p/35494575
        fmt ='.0%'#显示百分比
        fmt ='f' 显示完整数字 = fmt ='g'
        fmt ='.3'显示小数的位数 = fmt ='.3f' = fmt ='.3g'
        '''

        confusion_matrix = pd.DataFrame(data=disp.confusion_matrix, index=list(range(2)), columns=list(range(2)))

        #         f, ax = plt.subplots(figsize=(8,5),dpi=300)

        f, ax = plt.subplots(figsize=(8, 5), dpi=300)
        df_cm = pd.DataFrame(confusion_matrix)
        if i == 1:
            sns.heatmap(df_cm, annot=True, vmax=300.0, vmin=0.0, fmt='.0f', cmap='Blues', annot_kws={'size': 14})
        else:
            sns.heatmap(df_cm, annot=True, vmax=1.0, vmin=0.0, fmt='.2f', cmap='Blues', annot_kws={'size': 14})

        # label_x = ax.get_xticklabels()
        # label_y = ax.get_yticklabels()
        # plt.setp(label_x, rotation=45, horizontalalignment='right',fontsize=16)
        # plt.setp(label_y,  fontsize=16)

        scale_ls = np.array(range(2)) + 0.5
        index_ls = ['female', 'male']
        plt.xticks(scale_ls, index_ls)
        plt.yticks(scale_ls, index_ls)
        plt.xlabel('Predicted Lable', fontsize=20)
        plt.ylabel('True Lable', fontsize=20)
        plt.title(title)

        # plt.savefig('./fig_confusion_matrix_of_label_embedding_prediction.png')

        plt.show()

    # 计算
    # smiling positive
    # serious negative
    TN = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TP = matrix[1][1]
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * TP / (2 * TP + FP + FN)
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Error = (FP + FN) / (TP + FP + FN + TN)
    TNR = TN / (FP + TN)

    print('精确率(precision):', Precision, '\n', '召回率(Recall):', Recall, '\n', "F1:", F1, '\n', '准确率（Accuracy）:', Accuracy,
          '\n', '错误率(Error):', Error, '\n', '特异度(TNR):', TNR)

# LDA降维
X = np.load('X_lda_gender.npy')
y = np.load('y_lda_gender.npy')
class_names = list(['serious', 'smiling'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1666, random_state=0)
# clf1 = RandomForestClassifier(n_estimators=258,
#                               bootstrap=False,
#                               criterion='entropy',
#                               max_depth=16,
#                               max_features='sqrt',
#                               min_samples_leaf=4,
#                               min_samples_split=3).fit(X_train, y_train)
np.set_printoptions(precision=2)

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
lr = DecisionTreeClassifier(random_state=0)
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=lr)

for clf, label in zip([clf1, clf2, clf3, clf4, sclf],
                      ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'StackingClassifier']):
    clf.fit(X_train, y_train)
    print(label, '\n')
    i = 0
    # 绘制没有归一化的混淆矩阵
    titles_options = [("Confusion matrix (LDA+{name})".format(name=label), None),
                      ("Normalized confusion matrix (LDA+{name})".format(name=label), 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Greens,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        if i == 0:
            matrix = disp.confusion_matrix
        i += 1
        print(disp.confusion_matrix)
        import seaborn as sns

        sns.set(font_scale=1.4)  # y label 横向
        import matplotlib.pyplot as plt

        # fig=plt.figure(figsize=(8,5),dpi=400)
        '''
        https://zhuanlan.zhihu.com/p/35494575
        fmt ='.0%'#显示百分比
        fmt ='f' 显示完整数字 = fmt ='g'
        fmt ='.3'显示小数的位数 = fmt ='.3f' = fmt ='.3g'
        '''

        confusion_matrix = pd.DataFrame(data=disp.confusion_matrix, index=list(range(2)), columns=list(range(2)))

        #         f, ax = plt.subplots(figsize=(8,5),dpi=300)

        f, ax = plt.subplots(figsize=(8, 5), dpi=300)
        df_cm = pd.DataFrame(confusion_matrix)
        if i == 1:
            sns.heatmap(df_cm, annot=True, vmax=300.0, vmin=0.0, fmt='.0f', cmap='Blues', annot_kws={'size': 14})
        else:
            sns.heatmap(df_cm, annot=True, vmax=1.0, vmin=0.0, fmt='.2f', cmap='Blues', annot_kws={'size': 14})

        # label_x = ax.get_xticklabels()
        # label_y = ax.get_yticklabels()
        # plt.setp(label_x, rotation=45, horizontalalignment='right',fontsize=16)
        # plt.setp(label_y,  fontsize=16)

        scale_ls = np.array(range(2)) + 0.5
        index_ls = ['female', 'male']
        plt.xticks(scale_ls, index_ls)
        plt.yticks(scale_ls, index_ls)
        plt.xlabel('Predicted Lable', fontsize=20)
        plt.ylabel('True Lable', fontsize=20)
        plt.title(title)

        # plt.savefig('./fig_confusion_matrix_of_label_embedding_prediction.png')

        plt.show()

    # 计算
    # smiling positive
    # serious negative
    TN = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TP = matrix[1][1]
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * TP / (2 * TP + FP + FN)
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Error = (FP + FN) / (TP + FP + FN + TN)
    TNR = TN / (FP + TN)

    print('精确率(precision):', Precision, '\n', '召回率(Recall):', Recall, '\n', "F1:", F1, '\n', '准确率（Accuracy）:', Accuracy,
          '\n', '错误率(Error):', Error, '\n', '特异度(TNR):', TNR)
