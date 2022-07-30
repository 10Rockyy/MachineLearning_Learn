import numpy as np
from matplotlib import pyplot as plt

x=np.arange(0,2.1,0.05)
y=(np.sin((x-2))**2)*(np.e**(-(x**2)))
plt.title(r'$f(x) = sin^2(x - 2) e^{-x^2}$')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.plot(x,y,linewidth='3',color= 'r')
plt.show()

# 在绘制函数的时候，画了很多次图像都与给出的不一样，检查了很久才发现表达式少了个'*'