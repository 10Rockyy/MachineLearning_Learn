import numpy as np
ar0=np.arange(1,5)
ar0=np.diag(ar0,k=-1)
print(ar0)

#用diag函数，k为负数代表在对角线下方，k为正数代表在对角线上方