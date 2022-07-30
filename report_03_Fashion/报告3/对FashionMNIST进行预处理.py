# 导入第三方库
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# 导入FashionMNIST数据集
fashion_mnist=keras.datasets.fashion_mnist #从Keras导入数据集
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
print(np.shape(train_images))
print(np.shape(train_labels))
numberlabels_to_Chineselabels=['T恤/上衣','裤子','套头衫','连衣裙','外套',
               '凉鞋','衬衫','运动鞋','包','短靴'] #这里为了对应数据里的图像，所以我这里先设置一个列表，储存每一类标签的中文对应词

# 显示数据集
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
train_images=np.reshape(train_images,(60000,28*28))
test_images=np.reshape(test_images,(10000,28*28))
train_images=train_images/255.0
test_images=test_images/255.0
