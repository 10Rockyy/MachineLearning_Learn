#导入第三方库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import cv2 as cv

#导入读取数据
#加载数据
digits = load_digits()
#绘制图像
# fig = plt.figure(figsize=(6, 6))  # 设置图像的大小
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# 设置图像个数以及大小
# for i in range(64):
#     ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
#     ax.imshow(digits.images[i], cmap=plt.cm.binary) #此处cmap设置颜色，binary表示二值图像
#     # 给图像贴上标签值
#     ax.text(0, 7, str(digits.target[i]))
# plt.show()

#创建Sigmoid函数
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

#可视化错误分类图像函数
def visualize(dataset,i):
    #cv.imshow('img',dataset.images[i])
    #cv.waitKey()
    plt.imshow(dataset.images[i],cmap=plt.cm.binary) #显示该错误分类的图像
    plt.show()

#创建训练模型
def train_logistic(dataset,targets,times=2000,num=0,alpha=0.01,n=10):
#times为训练次数，默认为2000次;num为训练样本个数，可自设;alpha为学习速率，默认为0.01;n为多元回归的元数，由于此处是0-9，所以默认为10
    if num==0:
        num=int((1/2)*(np.shape(dataset)[0])) #num为训练样本个数，默认为1/2的样本总数
    train_samples=dataset[0:num,:] #存入前num个样本进入
    m=np.shape(dataset)[1] #获得一个样本维度
    weights=np.ones((n,m))  #因为对于多元回归，所以此处应创立10行以储存不同的weights
    b=np.ones(n)   #因为对于多元回归，此处应创立10行以储存不同的b
    for i in range(0,10): #开始循环训练,此处i代表数字
        target=np.copy(targets[:num]) #将训练样本的标签存入
        judge=target[:]
        for j in range(num):
            if i==target[j]: #判断target与哪个数字相同
                judge[j]=1   #储存判断结果
            else:
                judge[j]=0   #储存判断结果
        for k in range(times):
            data_index=list(range(num)) #创建一个sum个样本个数的数组，存进来每一个数据的下标
            rand_index=np.random.randint(0,len(data_index))#随机选取下标
            error=judge[rand_index]-sigmoid(sum(weights[i]*train_samples[rand_index])+b[i]) #梯度上升进行计算
            weights[i]+=alpha*error*train_samples[rand_index]
            b[i]+=alpha*error
            del(data_index[rand_index])
    return weights,b #返回了权值w和偏移b的十个对应数组

    
#多元模型预测
def dataset_predict(dataset_origin,dataset,targets,num_predict=100):
#num_predict为预测数据个数，默认为100个
    weights,b=train_logistic(dataset,targets)
    prdict_data=dataset[-1-num_predict:-1] #取数据集中最后num_predict个样本进行预测
    max_wb=np.empty(num_predict) #找到最大可能性的wb将储存在这里，现在它是随机生成的
    for i in range(num_predict): #循环次数等于预测样本数
        maybe=np.zeros(10) #每个数据对应10个可能性
        for j in range(10): #循环10种分类权值w和偏移b，然后取可能性最大的
            maybe[j]=sigmoid(sum(weights[j]*prdict_data[i])+b[j])#找出每个数据对应的10种可能性
        max_wb[i]=np.argmax(maybe) #找出最大可能性的w和b
    accurary,wrong=0,0
    for i in range(num_predict):
           if targets[-1-num_predict+i]==max_wb[i]:
               print(f'第{i+1}次预测,真值为{targets[-1-num_predict+i]},预测值为{max_wb[i]}')
               accurary+=1
           else:
               print(f'第{i+1}次预测,真值为{targets[-1-num_predict+i]},预测值为{max_wb[i]}    错')
               visualize(dataset_origin,-1-num_predict+i)
               wrong+=1

    print(f'本次多元预测模型,共预测{num_predict}个样本，总计正确{accurary}个，错误{wrong}个，正确率为{((accurary/num_predict)*100)}%')

data=digits.data
targets=digits.target
dataset_predict(digits,data,targets)

#在完成作业的时候，先查询了sklearn的digits数据集，知道其是Bunch类型，相当于一个字典。
#一共有'data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'几种Keys。每种keys对应不同的文件
#images为ndarray类型，保存8*8的图像，里面的元素是float64类型，共有1797张图片
#data为ndarray类型，将images按行展开成一行，共有1797行
#target为ndarray类型，指明每张图片的标签，也就是每张图片代表的数字
#target_names为ndarray类型，数据集中所有标签值

#其次是一开始处理完遇到的bug后，程序能够运行，但预测精准度只有10%左右;
#然后我又一步步检查代码，看了两次才发现是有个地方数组的问题，不应该直接赋值，而应该用np.copy