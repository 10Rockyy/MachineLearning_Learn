#Spectral Clustering(谱聚类)
#谱聚类（spectral clustering）是广泛使用的聚类算法，比起传统的K-Means算法，谱聚类对数据分布的适应性更强，聚类效果也很优秀，同时聚类的计算量也小很多，并且实现起来也不复杂。
#谱聚类是从图论中演化出来的算法，后来在聚类中得到了广泛的应用。它的主要思想是把所有的数据看做空间中的点，这些点之间可以用边连接起来。距离较远的两个点之间的边权重值较低，而距离较近的两个点之间的边权重值较高，通过对所有数据点组成的图进行切图，让切图后不同的子图间边权重和尽可能的低，而子图内的边权重和尽可能的高，从而达到聚类的目的。
#谱聚类算法的主要优点有：
# 1）谱聚类只需要数据之间的相似度矩阵，因此对于处理稀疏数据的聚类很有效。这点传统聚类算法比如K-Means很难做到
# 2）由于使用了降维，因此在处理高维数据聚类时的复杂度比传统聚类算法好。
# 谱聚类算法的主要缺点有：
# 1）如果最终聚类的维度非常高，则由于降维的幅度不够，谱聚类的运行速度和最后的聚类效果均不好。
# 2) 聚类效果依赖于相似矩阵，不同的相似矩阵得到的最终聚类效果可能很不同。

#导入第三方库
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import SpectralClustering

#读取数据并预处理
data=pd.read_csv('dataset_circles.csv',header=None,names=['x','y','labelnum']) #给数据添加分类标签
data.to_csv('dataset_circles_labeled.csv',index=False) #储存为新文件
data_new=pd.read_csv('dataset_circles_labeled.csv')
data=data_new
origin=data.values #转换为数组类型进行处理
xy=origin[:,:2] #提取x,y

#SC谱聚类算法
k=int(input('请输入K值：'))
sc=SpectralClustering(n_clusters=k,affinity='nearest_neighbors')
sc_cluster=sc.fit_predict(xy)
# print(sc_cluster)

# 可视化
color=['red','blue','gold','cyan','orchid','orange']
for i in range(len(xy)):
    plt.scatter(xy[i][0],xy[i][1],color=color[(sc_cluster[i])])  #根据分类数值不一样，绘制的颜色索引不一样
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.title('Spectral Clustering Display')
plt.show()

#DBSCAN(密度聚类算法)
#DBSCAN是一种基于密度的聚类算法，这类密度聚类算法一般假定类别可以通过样本分布的紧密程度决定。同一类别的样本，他们之间的紧密相连的，也就是说，在该类别任意样本周围不远处一定有同类别的样本存在。通过将紧密相连的样本划为一类，这样就得到了一个聚类类别。通过将所有各组紧密相连的样本划为各个不同的类别，则我们就得到了最终的所有聚类类别结果。
#形象来说，我们可以认为这是系统在众多样本点中随机选中一个，围绕这个被选中的样本点画一个圆，规定这个圆的半径以及圆内最少包含的样本点，如果在指定半径内有足够多的样本点在内，那么这个圆圈的圆心就转移到这个内部样本点，继续去圈附近其它的样本点。
#等到这个圈发现所圈住的样本点数量少于预先指定的值，就停止了。那么我们称最开始那个点为核心点，停下来的那个点为边界点，没有在圈里的那个点为离群点。
#参数：
# eps：epsilon,圈半径
# min_samples:圈内圈住个数

#导入第三方库
import pandas as pd
from sklearn.cluster import DBSCAN

#读取数据预处理
data=pd.read_csv('dataset_circles.csv',header=None,names=['x','y','labelnum'])

#DBSCAN算法
db=DBSCAN(eps=6,min_samples=6).fit(data)
labelnum=db.labels_  #将分类的标签生成为列表
#print(labelnum) #For test

# 可视化
color=['orchid','orange','red','blue','gold','cyan']
for i in range(len(xy)):
    plt.scatter(xy[i][0],xy[i][1],color=color[(sc_cluster[i])])  #根据分类数值不一样，绘制的颜色索引不一样
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.title('DBSCAN Display')
plt.show()







