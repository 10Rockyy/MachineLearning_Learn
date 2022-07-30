import numpy as np
# 3x + 4y + 2z = 10
# 5x + 3y + 4z = 14
# 8x + 2y + 7z = 20

#用内置函数solve解决，需注意等式右边矩阵br应为列向量
a1=[[3,4,2],[5,3,4],[8,2,7]]
b1=[10,14,20]
ar1=np.array(a1)
br=np.array(b1)
r1=np.linalg.solve(ar1,br.T)
print(r1)

#运用数学原理，X=A^(-1)B，所以需要先求A的逆矩阵，再用A的逆矩阵点乘B
a2=[[3,4,2],[5,3,4],[8,2,7]]
b2=[10,14,20]
ar2=np.linalg.inv(a2)
r2=ar2.dot(b2)
print(r2)