import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import math

def visualise_data(data):
    plt.axis([-50,50,-50,50])
    ar=np.shape(data)
    for i in range(ar[0]):
        if data['labelnum'][i]==0.0:
            plt.plot(data['x'][i],data['y'][i],'ro',markersize=2)
        else:
            plt.plot(data['x'][i],data['y'][i],'bo',markersize=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.title('Origin_data Display')
    plt.show()

def origin_location(data,k):
    origin_location_new=np.array([]) #初始化重心的数组
    for i in range(k): #随机获取重心，并将其连接为一个数组
        origin_location=data[random.randint(0,m)]
        origin_location_new=np.concatenate((origin_location,origin_location_new),axis=0)
    origin_location_new=origin_location_new.reshape(-1,2) #将数组形状变为列为2
    print(f'随机选取的初始重心为：\n{origin_location_new}\n')
    return origin_location_new

def distance(location,dot): #计算该点与各重心之间的欧式距离,并返回一个k行数组
    dist=(location-dot)**2
    dist_new=np.sum(dist,axis=1) #将每一行的值平方和相加，即得到该点与每个重心的距离平方和
    return dist_new

def cluster(location,data):
    m,n=np.shape(data)
    cluster_init=np.mat(np.zeros((m,2)))
    flags=1 #做标记
    num=0 #计算的次数
    while flags:
        flags=0
        for i in range(0,m): #对每个样本进行循环
            distance_new=distance(location,data[i]) #对每个点求欧式距离
            distance_k=np.argmin(distance_new) #找到该点属于哪一个重心类
            if distance_k != cluster_init[i,0]:
                flags=1
            cluster_init[i][0]=distance_k
        num=num+1
        for j in range(0,k): #找到最新的聚类重心
            location_new=data[np.nonzero(cluster_init[:,0].A==j)[0]] #已知每一样本的类，计算最新的聚类重心
            location[j,:]=np.mean(location_new,axis=0)
    return location,num,cluster_init

def display(origin_data,cluster_data):
    visualise_data(origin_data) #原数据

    plt.axis([-50,50,-50,50]) #聚类
    color=['red','blue','gold','cyan','orchid','orange']
    for i in range(m):
        nn= int(cluster_data[i,0])
        plt.plot(xy[i,0],xy[i,1],'o', markersize=2,color=color[nn])
    for j in range(k):
        plt.plot(location[j,0],location[j,1],'*',markersize=16,color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.title('kMeans Display')
    plt.show()

k=int(input('请输入K值：'))
data=pd.read_csv('dataset_circles.csv',header=None,names=['x','y','labelnum']) #给数据添加分类标签
data.to_csv('dataset_circles_labeled.csv',index=False)  #储存为新文件
data_new=pd.read_csv('dataset_circles_labeled.csv')
data=data_new
origin=data.values #转换为数组类型进行处理
xy=origin[:,:2] #提取x,y
m,n=np.shape(xy) #获得每个维度个数
location=origin_location(xy,k)
location,num,cluster_data=cluster(location,xy)
print(f'聚类重心为：\n{location}\n一共计算了{num}次')
display(data,cluster_data)

