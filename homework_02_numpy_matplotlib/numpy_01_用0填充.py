import numpy as np
ar=np.array([[10,34,54,23],[31, 87, 53, 68],[98, 49, 25, 11],[84, 32, 67, 88]])
ar=np.pad(ar , pad_width=1 , mode='constant' , constant_values=0)
print(ar)

#利用pad函数直接进行填充，填充方法使用constant,即可沿着边缘填充，厚度可自行给定