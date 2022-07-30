# 过程
# 先使用上次自己写的NN来训练模型，预测样本
# 再使用sklearn的MLPClassifier

import NN
import pandas as pd
import time
from sklearn.metrics import roc_auc_score
import numpy as np

# 预处理数据
train_dataset=pd.read_csv('data/train.csv') #导入训练数据集
predict_dataset=pd.read_csv('data/test.csv') #导入测试数据集
train_labels=train_dataset['Evaluation'] #保存测试训练集样本标签
# print(type(train_labels))
train_dataset=train_dataset.drop(['CaseId','Evaluation'],axis=1) #预处理，去掉不需要的CaseId和Evaluation列
predict_dataset=predict_dataset.drop(['CaseId'],axis=1) #预处理，去掉不需要的CaseId列
train_labels=train_labels.to_numpy() #从Dataframe格式转为ndarray格式
train_dataset=train_dataset.to_numpy() #从Dataframe格式转为ndarray格式
# train_dataset=np.array(train_dataset)
# train_labels=np.array(train_labels)
# print(np.shape(train_dataset))

nn=NN.NN(train_dataset[:2000],train_labels[:2000]) #此处因为自己写的NN网络过于菜，所以这里只训练了2000个样本，多了属实用时太长
nn.change_Parameters(input_num=36,hide_num=2,output_num=2,alpha=0.0026,times=1000) #这里是输入36个维度，隐层2个维度，输出2个维度，alpha为试出来感觉比较好的值，计算1000次
time_start=time.time()# 计时
nn.init_wb() #初始化
nn.back_calculate() #反向传播计算
time_end=time.time() #结束计时
print(f'训练用时{time_end-time_start}s')
train_dataset_predictlabels=nn.predict(train_dataset,train_labels) #预测


# 计算AUC值
AUC=roc_auc_score(train_labels,train_dataset_predictlabels) #直接调用函数计算
print(f'自己写的NN模型的AUC值为{AUC}')

# 保存测试数据集的预测结果
predict_dataset_predictlabels=nn.predict_no_orgin_labels(predict_dataset,[0]) #预测测试数据集样本
Caseid=pd.DataFrame(np.array(np.arange(80000)+200001),columns=['Caseid']) #生成id数组，并且生成文件类型的列
predictlabels=pd.DataFrame(predict_dataset_predictlabels,columns=['Evaluation']) #将预测的标签生成文件类型的列
submit_data_LR=pd.concat([Caseid,predictlabels],axis=1) #将id与预测标签合并
submit_data_LR.to_csv("submit_data/submit_data_自己写的NN",index=False) #生成的文件保存

# exp函数溢出
# 边际效应递减






