import numpy as np
from matplotlib import pyplot as plt

#一维
plt.subplot(1,2,1)
np.random.seed()
n=int(input('请输入醉汉走步数量：'))
x=np.arange(0,n)
y=np.random.rand(n)
plt.title('one-dimension')
plt.grid(axis='x')
plt.plot(x,y)

#二维
plt.subplot(1,2,2)
np.random.seed()
x=np.arange(n)
y=n*(np.random.rand(n))
plt.scatter(x[0],y[0],c='r',s=20)
plt.scatter(x[-1],y[-1],c='y',s=80)
plt.title('two-dimension')
plt.plot(x,y,c='r')
plt.grid()

#三维
fig=plt.figure()
a= fig.gca(projection='3d')
z=np.arange(n)
x=n*(np.random.rand(n))
y=n*(np.random.rand(n))
a.scatter(x[0],y[0],z[0],c='r',s=20)
a.scatter(x[-1],y[-1],z[-1],c='y',s=80)
a.view_init(elev=6, azim=20)
plt.title('three-dimension')
a.plot(x,y,z)
plt.show()


# 增加了网格以区分一维和二维，另外将二维的x,y轴的数值大小范围设置为一致，以表示醉汉的位置
# 虽然三维的我试过很多种视角，但是感觉还是不能看得很清楚