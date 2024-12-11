# _*_ coding : utf-8 _*_
# @Time : 2024/3/11 22:04
# @Author : Li
# @File : lod_roc2
# @Project : method4
# _*_ coding : utf-8 _*_
# @Time : 2024/3/10 13:59
# @Author : Li
# @File : load_roc
# @Project : method4
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
# import pandas as pd

import pandas as pd
from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 从 CSV 文件读取数据
data1 = pd.read_csv('./roc_fanxiu/books_label_score_data_1.csv')
data2 = pd.read_csv('./roc_fanxiu/books_label_score_data_2.csv')
data3 = pd.read_csv('./roc_fanxiu/books_label_score_data_4.csv')
data4 = pd.read_csv('./roc_fanxiu/books_label_score_data_5.csv')
# 提取标签和异常得分
# 提取标签和异常得分
y_true1 = data1['Label']
scores1 = data1['Anomaly Score']

y_true2 = data2['Label']
scores2 = data2['Anomaly Score']

y_true3 = data3['Label']
scores3 = data3['Anomaly Score']

y_true4 = data4['Label']
scores4 = data4['Anomaly Score']

# 计算 ROC 曲线
fpr1, tpr1, _ = roc_curve(y_true1, scores1)
fpr2, tpr2, _ = roc_curve(y_true2, scores2)
fpr3, tpr3, _ = roc_curve(y_true3, scores3)
fpr4, tpr4, _ = roc_curve(y_true4, scores4)
# 计算 AUC
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)
roc_auc4 = auc(fpr4, tpr4)
# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr1, tpr1, color='red', lw=1, label='1(area = %0.4f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='green', lw=1, label='2(area = %0.4f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='blue', lw=1, label='4 (area = %0.4f)' % roc_auc3)
plt.plot(fpr4, tpr4, color='blue', lw=1, label='5 (area = %0.4f)' % roc_auc4)

# plt.plot(fpr1, tpr1, color='red', lw=1, label='1')
# plt.plot(fpr2, tpr2, color='green', lw=1, label='2')
# plt.plot(fpr3, tpr3, color='blue', lw=1, label='3')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('', loc='center')
plt.legend(loc="lower right")
plt.show()