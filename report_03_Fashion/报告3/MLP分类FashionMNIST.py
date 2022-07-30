from sklearn.neural_network import MLPClassifier
import time
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib
from matplotlib import pyplot as plt

# 导入和预处理数据集
fashion_mnist=keras.datasets.fashion_mnist #从Keras导入数据集
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
train_images=np.reshape(train_images,(60000,28*28))
test_images=np.reshape(test_images,(10000,28*28))
train_images=train_images/255.0
test_images=test_images/255.0
numberlabels_to_Chineselabels=['T恤/上衣','裤子','套头衫','连衣裙','外套',
               '凉鞋','衬衫','运动鞋','包','短靴'] #这里为了对应数据里的图像，所以我这里先设置一个列表，储存每一类标签的中文对应词


# MLP训练
mlp=MLPClassifier(max_iter=1000)
mlp_x_train=train_images[:] #生成训练数据集样本
mlp_y_train=train_labels[:] #生成训练数据集标签
time_start=time.time() #计时
mlp.fit(mlp_x_train,mlp_y_train) #训练
time_end=time.time() #结束计时
print(f'训练用时{time_end-time_start}s')

# 预测(训练数据集)
mlp_predict=mlp.predict(test_images) #预测
train_dataset_accuracy=accuracy_score(test_labels,mlp_predict) #得到训练数据集的预测正确率
print(f'使用训练数据集测试MLP模型，正确率为{train_dataset_accuracy*100}%')

# 可视化结果展示
def plot_image(i,predict_label,true_label,img):
    img=np.reshape(img,(10000,28,28)) #现将img转换为28*28的数组
    predict_label,true_label,img=predict_label[i],true_label[i],img[i]
    plt.imshow(img,cmap=plt.cm.binary) #绘制出img
    if predict_label==true_label:
        plt.xlabel(f'{numberlabels_to_Chineselabels[predict_label]}(正确)',
                   fontproperties='Heiti TC',size=16,color='blue') #正确
    else:
        plt.xlabel(f'{numberlabels_to_Chineselabels[predict_label]}(错误)',
                   fontproperties='Heiti TC',size=16,color='red') #错误
# i代表第几个图像，predict_label为预测得到的标签，true_label为真实的标签，img为图像集
plot_image(i=0,predict_label=mlp_predict,true_label=test_labels,img=test_images)

# 利用softmax看每一个标签的概率
probability=mlp.predict_proba(test_images) #得到每个标签的概率
def plot_probability(i,probability,true_label):
    probability,true_label=probability[i],true_label[i]
    plt.xticks(range(10)) #x轴刻度范围
    bar=plt.bar(range(10),probability,color='red')
    plt.ylim([0,1])
    predict_label=np.argmax(probability) #概率最高的为预测标签
    bar[predict_label].set_color('red')
    bar[true_label].set_color('blue')
# i代表第几个图像，probability为每种标签的概率，true_label为真实的标签
plot_probability(i=0,probability=probability,true_label=test_labels)

# 预测一堆图像
def plot_many(row,column,predict_label,true_label,img,probability):
    num=row*column  #总共要绘制图片数
    plt.figure(figsize=(2*2*column,2*row)) #先将所有图片数量定义好
    for i in range(num):
        plt.subplot(row,2*column,2*i+1)
        plot_image(i,predict_label,true_label,img)
        plt.subplot(row,2*column,2*i+2)
        plot_probability(i,probability,true_label)
    plt.tight_layout()
    plt.show()
# row代表行，column代表列，predict_label为预测得到的标签，true_label为真实的标签，img为图像集，probability为每种标签的概率
plot_many(row=3,column=3,predict_label=mlp_predict,
          true_label=test_labels,img=test_images,probability=probability)