import numpy as np
ar=np.arange(1,65).reshape(8,8)
ls=ar.shape
#print(type(ls))   #获得函数返回类型
n=len(ls)         #获得数组维度
for i in range(n):
    ar=np.flip(ar,axis=i)
print(ar)

#可对任意维度数组进行翻转
#此处为8x8数组