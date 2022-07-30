import tensorflow as tf
from tensorflow import keras
import time
import numpy as np

# 结果放入models中

# 数据集加载函数，指明数据集的位置并统一处理为imgheight*imgwidth的大小，同时设置batch

# CNN模型训练函数，训练好的模型保存在工程文件下的'models'目录下
# 构建CNN模型
def CNN(class_num,IMG_SHAPE=(28,28,1)):
    # 搭建模型
    model=tf.keras.models.Sequential([
        # 对模型做归一化的处理，将0-255之间的数字统一处理到0到1之间
        # 归一化处理可以减少计算量，并且提高训练出的模型精度
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255,input_shape=IMG_SHAPE),
        # 卷积层，该卷积层的输出为32个通道
        # 此处使用的卷积核的大小是3*3，激活函数使用relu
        tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu'),
        # 池化层，此处使用的kernel大小是2*2
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        # 增添另一个卷积池化
        # 卷积层，该卷积层的输出为64个通道
        # 此处使用的卷积核的大小是3*3，激活函数使用relu
        tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
        # 池化层，此处使用的kernel大小是2*2
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        # 拉直层，将二维的输出转化为一维
        tf.keras.layers.Flatten(input_shape=IMG_SHAPE),
        # 全连接层，输出值维度为128，激活函数用的'relu'
        tf.keras.layers.Dense(128,activation='relu'),
        # 利用softmax函数的概率值进行分类
        tf.keras.layers.Dense(class_num,activation='softmax')])
    # 输出模型信息
    model.summary()
    # 配置模型训练方法，sgd优化器，损失函数为交叉熵函数，模型评价指标为正确率
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    # 返回模型
    return model

def train(train_img,train_labels,epochs=30):
    # train需传入参数：训练集、验证集文件目录，epoch为循环次数可以修改，默认30次
    # 开始训练，记录开始时间
    time_start=time.time()
    # 创建CNN模型
    model=CNN(class_num=len(train_labels))
    # 训练
    model.fit(train_img,train_labels,epochs=epochs)
    # 保存训练好的模型以备后续使用
    model.save("TF训练好的models/CNN_FashionMNIST.h5")
    # 记录结束时间
    end_time=time.time()
    print(f'训练的总时间用时为：{end_time-time_start}秒')

# 导入数据集
fashion_mnist=keras.datasets.fashion_mnist  # 从Keras导入数据集
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
# 这里为了对应数据里的图像，所以我这里先设置一个列表，储存每一类标签的中文对应词
numberlabels_to_Chineselabels=['T恤/上衣','裤子','套头衫','连衣裙','外套',
                                 '凉鞋','衬衫','运动鞋','包','短靴']
train_images=np.expand_dims(train_images,axis=3) #这里需要将图像纬度进行更改，不然训练CNN的时候会报错（说的是数据少了一个维度）
# print(np.shape(train_images))
# train需传入参数：训练集、以及对应的标签，epoch为循环次数可以修改，默认30次
train(train_img=train_images,train_labels=train_labels)

