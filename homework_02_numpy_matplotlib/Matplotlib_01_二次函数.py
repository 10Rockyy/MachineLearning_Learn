import numpy as np
from matplotlib import pyplot as plt
def hanshu():
    y=input('请输入函数(不用输入y，例如直接输入x**2)：')
    y1=y.replace('x','x1')
    return y,y1

x=np.arange(-10,10,0.1)
y,y1=hanshu()
y=eval(y)
plt.plot(x,y)    #绘制函数

n=int(input('请输入等分份数：'))
x2=np.arange(-10,10.1,20/n)
for i in x2:     #绘制等分梯形
    x1=np.array([i,i])
    y2=eval(y1)
    a=y2[0]
    y3=np.array([0,a])
    plt.plot(x1,y3,linewidth='0.5')
plt.show()

# 梯形是用两点(x1,0)与(x1,y)两点绘制的直线
# 其中绘图部分比较好解决，但是在引入直接输入函数部分比较困难，花了比较多的时间用于解决。
# 因为要用两次输入的函数，但是不能使用户输入两次函数，所以想到了用replace代替自变量，使其变为两个函数
# 因为第一次是用这个库画图，所以这道题用时还挺久的，不过对于函数以及方法还比较熟悉了