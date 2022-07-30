import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import math

k=int(input('请输入K值：'))
data=pd.read_csv('dataset_circles.csv',header=None,names=['x','y','labelnum']) #给数据添加分类标签
data.to_csv('dataset_circles_labeled.csv',index=False)
data_new=pd.read_csv('dataset_circles_labeled.csv')
data=data_new

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

def polar_coordinate(data):
    m,n=np.shape(data) #获取每个维度长度
    ar=np.zeros((m,n))#创建m*n的空数组
    for i in range(m):
        r=np.sqrt(((data[i][0])**2) + ((data[i][1])**2))  #获得每个点与原点的距离
        theta=np.math.atan(data[i][1]/data[i][0]) #获得极坐标的theta角
        ar[i,:]=r,theta
    return ar

def distance(location ,dot):
    dist=(location-dot)**2
    dist_new=np.sum(dist,axis=1)
    return dist_new

def origin_location(xy,k):
    origin_location_new=np.array([]) #初始化重心的数组
    for i in range(k): #随机获取重心，并将其连接为一个数组
        origin_location=xy[random.randint(0,m)]
        origin_location_new=np.concatenate((origin_location,origin_location_new),axis=0)
    origin_location_new=origin_location_new.reshape(-1,2) #将数组形状变为列为2
    print(f'随机选取的初始重心为：\n{origin_location_new}\n')
    return origin_location_new

def cluster_polar(data,k,location_polar):
    m,n=np.shape(data)
    cluster_init=np.mat(np.zeros((m,2)))
    flags=1
    num=0
    while flags:
        flags=0
        for i in range(0,m):
            distance_new=distance(location_polar,data[i]) #对每个点求欧式距离
            distance_k=np.argmin(distance_new) #找到该点属于哪一个重心类
            if distance_k!=cluster_init[i,0]:
                flags =1
            cluster_init[i][0]=distance_k
        num+=1
        for j in range(k):
            location_new=data[np.nonzero(cluster_init[:,0].A==j)[0]] #已知每一样本的类，计算最新的聚类重心
            location_polar[j,:]=np.mean(location_new,axis=0)
    return location_polar,cluster_init,num

def display(origin_data,cluster_data):
    visualise_data(origin_data)#原数据

    plt.axis([-50,50,-50,50]) #聚类
    color=['red','blue','gold','cyan','orchid','orange']
    for i in range(m):
        nn= int(cluster_data[i,0])
        plt.plot(xy[i,0],xy[i,1],'o', markersize=2,color=color[nn])
    for j in range(k):
        plt.plot(location_polar[j,0],location_polar[j,1],'*',markersize=16,color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.title('Polar_cluster Display')
    plt.show()

origin=data.values #转换为数组类型计算
xy=origin[:,:2] #提取x,y
m,n=np.shape(data)
data_polar=polar_coordinate(xy) #进行极坐标变换（利用上述定义函数）
location_polar=origin_location(data_polar,k) #获得随机初始的重心点(注：此处用变换后的data_polar)
location_polar,cluster_data,num=cluster_polar(data_polar,k,location_polar)
print(f'聚类重心极坐标为：\n{location_polar}\n一共计算了{num}次')
display(data,cluster_data)
