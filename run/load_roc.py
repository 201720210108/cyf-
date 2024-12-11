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
data1 = pd.read_csv('./roc/eat_label_score_data.csv')
data2 = pd.read_csv('./roc/eat_label_score_data_noDA.csv')
data3 = pd.read_csv('./roc/eat_label_score_data_noFC.csv')
# 提取标签和异常得分
# 提取标签和异常得分
y_true1, scores1 = data1['Label'], data1['Anomaly Score']
y_true2, scores2 = data2['Label'], data2['Anomaly Score']
y_true3, scores3 = data3['Label'], data3['Anomaly Score']

# 计算三组 ROC 曲线
fpr1, tpr1, _ = roc_curve(y_true1, scores1)
fpr2, tpr2, _ = roc_curve(y_true2, scores2)
fpr3, tpr3, _ = roc_curve(y_true3, scores3)

# 去除重复的数据点
fpr1_unique, indices = np.unique(fpr1, return_index=True)
tpr1_unique = tpr1[indices]

fpr2_unique, indices = np.unique(fpr2, return_index=True)
tpr2_unique = tpr2[indices]

fpr3_unique, indices = np.unique(fpr3, return_index=True)
tpr3_unique = tpr3[indices]

# 计算 AUC
roc_auc1 = auc(fpr1_unique, tpr1_unique)
roc_auc2 = auc(fpr2_unique, tpr2_unique)
roc_auc3 = auc(fpr3_unique, tpr3_unique)

# 绘制 ROC 曲线
plt.figure()

# 绘制原始曲线
plt.plot(fpr1_unique, tpr1_unique, color='darkorange', lw=2, label='ROC curve 1 (area = %0.2f)' % roc_auc1)
plt.plot(fpr2_unique, tpr2_unique, color='blue', lw=2, label='ROC curve 2 (area = %0.2f)' % roc_auc2)
plt.plot(fpr3_unique, tpr3_unique, color='green', lw=2, label='ROC curve 3 (area = %0.2f)' % roc_auc3)

# 进行样条插值并绘制平滑曲线
fpr_interp = np.linspace(0, 1, 1000)  # 用于插值的新的 fpr 值

# 对每条曲线进行插值
tpr1_interp = interp1d(fpr1_unique, tpr1_unique, kind='cubic', assume_sorted=True)
tpr2_interp = interp1d(fpr2_unique, tpr2_unique, kind='cubic', assume_sorted=True)
tpr3_interp = interp1d(fpr3_unique, tpr3_unique, kind='cubic', assume_sorted=True)

# 绘制平滑后的曲线
# plt.plot(fpr_interp, tpr1_interp(fpr_interp), color='darkorange', lw=2, linestyle='--', label='Smoothed ROC curve 1')
# plt.plot(fpr_interp, tpr2_interp(fpr_interp), color='blue', lw=2, linestyle='--', label='Smoothed ROC curve 2')
# plt.plot(fpr_interp, tpr3_interp(fpr_interp), color='green', lw=2, linestyle='--', label='Smoothed ROC curve 3')
# plt.plot(fpr_interp, tpr1_interp(fpr_interp), color='darkorange', lw=2, label='Smoothed ROC curve 1')
# plt.plot(fpr_interp, tpr2_interp(fpr_interp), color='blue', lw=2, label='Smoothed ROC curve 2')
# plt.plot(fpr_interp, tpr3_interp(fpr_interp), color='green', lw=2,  label='Smoothed ROC curve 3')

# 绘制对角线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 设置图形属性
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()