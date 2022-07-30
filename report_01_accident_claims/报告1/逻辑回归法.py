# 思路
# 先利用pandas对数据集进行预处理
# 利用sklearn的逻辑回归库对数据进行学习
# 对数据进行预测
# 绘制ROC曲线，计算预测准确度、AUC值等

# 导入第三方库
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
import time

# 预处理数据
train_dataset=pd.read_csv('data/train.csv') #导入训练数据集
predict_dataset=pd.read_csv('data/test.csv') #导入测试数据集
train_labels=train_dataset['Evaluation'] #保存测试训练集样本标签
# print(type(train_labels))
train_dataset=train_dataset.drop(['CaseId','Evaluation'],axis=1) #预处理，去掉不需要的CaseId和Evaluation列
predict_dataset=predict_dataset.drop(['CaseId'],axis=1) #预处理，去掉不需要的CaseId列

# 逻辑回归算法
time_start=time.time()
LR=LogisticRegression(max_iter=5000) #生成逻辑回归实例
LR.fit(train_dataset,train_labels) #训练
print(f'训练用时{time.time()-time_start}s')
# 用训练数据集预测
time_start=time.time()
train_dataset_predictlabels=LR.predict(train_dataset) #得到训练数据集的预测标签
print(f'预测训练集的样本用时{time.time()-time_start}s')
train_dataset_accuracy=accuracy_score(train_labels,train_dataset_predictlabels) #得到训练数据集的预测正确率
print(f'使用训练数据集测试LR模型，正确率为{train_dataset_accuracy*100}%')
# 用测试数据集预测
predict_dataset_predictlabels=LR.predict(predict_dataset) #预测
# 保存测试数据集的预测结果
Caseid=pd.DataFrame(np.array(np.arange(80000)+200001),columns=['Caseid']) #生成id数组，并且生成文件类型的列
predictlabels=pd.DataFrame(predict_dataset_predictlabels,columns=['Evaluation']) #将预测的标签生成文件类型的列
submit_data_LR=pd.concat([Caseid,predictlabels],axis=1) #将id与预测标签合并
submit_data_LR.to_csv("submit_data/submit_data_LR",index=False) #生成的文件保存

# 绘制ROC曲线(因为没有测试数据集的真正标签，所以用训练数据集的数据进行ROC曲线绘制，并且计算AUC)
predictlabels_to_one=LR.predict_proba(train_dataset)[:,1] #得到训练数据集预测标签为1的概率
FPR,TRP,thresholds=roc_curve(train_labels,predictlabels_to_one) #获得真正率或假正率
plt.figure(1) #绘图
plt.plot(FPR,TRP) #绘制ROC曲线
plt.xlim([0,1]) #x范围
plt.ylim([0,1]) #y范围
plt.plot([0,1],[0,1],'--') #随机分布曲线
plt.title('ROC Curve') #标题
plt.grid() #网格（更好看）
plt.xlabel('FPR') #横坐标
plt.ylabel('TPR') #纵坐标
plt.show() #show

# 绘制PR曲线
Precision,Recall,thresholds=precision_recall_curve(train_labels,predictlabels_to_one) #通过调用函数获得精确和召回
plt.figure(2) #绘图
plt.plot(Precision,Recall) #绘制PR曲线
plt.xlim([0,1]) #x范围
plt.ylim([0,1]) #y范围
plt.title('PR Curve') #标题
plt.grid() #网格（更好看）
plt.xlabel('Recall') #横坐标
plt.ylabel('Precision') #纵坐标
plt.show() #show

# 计算AUC值
AUC=roc_auc_score(train_labels,train_dataset_predictlabels) #直接调用函数计算
print(f'LR模型的AUC值为{AUC}')



#运用pandas储存文件时遇到问题