import numpy as np

ar=np.ones((8,8),dtype=int)
ar[::2,1:9:2]=0  #先作用于列
ar[1:9:2,::2]=0  #再作用于行
print(ar)

#通过先固定行，将所有列变换，然后再将所有行变换，其中每次变换的步长为2