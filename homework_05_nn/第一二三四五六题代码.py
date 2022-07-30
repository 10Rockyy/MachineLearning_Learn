#本题为搭建一个2层的神经网络，其中包含1个隐层和1个输出层
#为了方便起见，我打算直接把第(1)(2)(3)小问一起做，直接面向对象封装成类，然后直接做成对各种数据集都适用的NN
#预测结果的可视化直接用预测结果与原数据分类对比

#导入第三方库
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import time

#加载原数据
np.random.seed(0)
x,y=datasets.make_moons(200, noise=0.20) #生成月亮图形状数据,其中x代表数据坐标,y代表数据分类
#plt.scatter(x[:, 0],x[:, 1],c=y,cmap=plt.cm.Spectral)
#plt.show()

#创建Sigmoid函数
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

#创建NN的类
class NN:
    def __init__(self,datasets,type): #datasets表示训练样本数据集，type代表训练样本分类
        self.alpha=0.01 #学习速率
        self.times=1000 #计算次数
        self.input_num=2 #输入层维度
        self.hide_num=2 #隐层维度数
        self.output_num=2 #输出层维度
        self.datasets=datasets #训练或预测数据集
        self.type=type #数据集内数据对应类别

    #传入模型参数(设置此函数的目的，是为了应对不同数据时，使模型的参数可变)
    def change_Parameters(self,input_num,hide_num,output_num,alpha,times):
        self.input_num=input_num
        self.hide_num=hide_num
        self.output_num=output_num
        self.alpha=alpha
        self.times=times

    #随机初始化权重w与偏移b
    def init_wb(self):
        self.w1,self.b1,self.w2,self.b2=[],[],[],[]  #创建w,b矩阵
        n1,n2=np.sqrt(self.input_num),np.sqrt(self.hide_num) #创建参数n1,n2便于后续计算
        self.w1=np.random.randn(self.input_num,self.hide_num)/n1  #根据输入维度,隐层维度输出生成w1,b1矩阵
        self.b1=np.zeros((1,self.hide_num)) #生成b1为(1,隐层维度)的矩阵，其中矩阵内的数值随机生成
        self.w2=np.random.randn(self.hide_num, self.output_num)/n2  #根据输入维度,隐层维度输出生成w2,b2矩阵
        self.b2=np.zeros((1,self.output_num))  #生成b2为(1,输出层维度)的矩阵，其中矩阵内的数值随机生成
        #将原始数据类型储存
        self.true_type=np.zeros((self.datasets.shape[0],self.output_num))  #初始化数据类别矩阵
        for i in range(self.output_num):
            self.true_type[np.where(self.type==i),i]=1 #将数据类别矩阵进行填充，用原始数据的类别

    #定义向前计算函数
    def forward_calculate(self):
        self.z1=sigmoid(self.datasets.dot(self.w1)+self.b1)  #对应数学表达式：z1=sigmoid(X*w1+b1)，为隐层的输出
        self.z2=sigmoid(self.z1.dot(self.w2)+self.b2)  #对应数学表达式：z2=sigmoid(X*w2+b2)，为输出层输出

    #定义反向计算传播函数
    def back_calculate(self):
        for i in range(self.times): #一共反向计算的次数
            self.forward_calculate() #进行向前运算
            predict_result=np.argmax(self.z2,axis=1) #寻找最大项来作为预测值
            self.accuracy=accuracy_score(self.type,predict_result) #调用sklearn计算准确度
            print(f'这是第{i+1}次训练,正确率为{self.accuracy*100}%')
            if self.accuracy==1:
                return
            #每一次正向计算后的结果和真实结果做比较，计算误差值；
            #然后将误差引入w和b，根据误差函数值改变w和b；
            #然后重复进行上述步骤，直到达到计算次数或没有误差。
            n2=self.z2*(1-self.z2)*(self.true_type-self.z2)
            n1=self.z1*(1-self.z1)*(np.dot(n2,self.w2.T))
            self.w2=self.w2+self.alpha*np.dot(self.z1.T,n2)
            self.b2=self.b2+self.alpha*np.sum(n2,axis=0)
            self.w1=self.w1+self.alpha*np.dot(self.datasets.T,n1)
            self.b1=self.b1+self.alpha*np.sum(n1,axis=0)

    def predict(self,predict_datasets,predict_datasets_type):
        self.datasets=predict_datasets
        self.forward_calculate()  # 利用训练好的w,b进行向前运算
        predict_result=np.argmax(self.z2, axis=1)
        self.predict_result=predict_result
        self.predict_datasets=predict_datasets
        self.predict_datasets_type=predict_datasets_type
        print(f'此次预测数据个数为{np.shape(predict_datasets)[0]}个,预计预测准确度为{self.accuracy*100}%')
        accuracy=0
        for i in range(np.shape(predict_datasets)[0]): #将预测数据集中的每个数据都进行预测
            if predict_datasets_type[i]==predict_result[i]:
                accuracy+=1
                print(f'这是对第{i+1}个数据的预测，真实类别为{predict_datasets_type[i]}，预测类别为{predict_result[i]}')
            else:
                print(f'这是对第{i+1}个数据的预测，真实类别为{predict_datasets_type[i]}，预测类别为{predict_result[i]}           错')
        print(f'此次预测个数为{np.shape(predict_datasets)[0]}个，正确率为{(accuracy/np.shape(predict_datasets)[0])*100}%')

    def Visulise(self):
        plt.scatter(self.predict_datasets[:, 0],self.predict_datasets[:, 1],c=self.predict_datasets_type,cmap=plt.cm.Spectral) #原数据分类
        plt.title('Origin type')
        plt.show()
        plt.scatter(self.predict_datasets[:, 0],self.predict_datasets[:, 1],c=self.predict_result,cmap=plt.cm.Spectral) #预测数据分类
        plt.title('Predict data')
        plt.show()

    def softmax(self):
        #print(self.z2)
        softmax_result=np.zeros((np.shape(self.predict_datasets)[0],self.output_num)) #初始化softmax分类概率矩阵，其中维度按照(个数，类别)进行生成
        softmax_num=np.sum(np.exp(self.z2), axis=1) #让所有分类的概率先相加
        for i in range(np.shape(self.z2)[0]):
            for j in range(np.shape(self.z2)[1]):
                softmax_result[i,j]=np.exp(self.z2[i,j])/softmax_num[i] #用定义，计算每一类的值占总值的比例
        for i in range(np.shape(self.predict_datasets)[0]):
            if self.predict_datasets_type[i]==self.predict_result[i]:
                print(f'这是第{i+1}个数据的Softmax分类结果：\n'
                      f'{softmax_result[i]}\n'
                      f'真实类别是{self.predict_datasets_type[i]},概率最大的为第{self.predict_result[i]+1}个，代表{self.predict_result[i]}\n')
            else:
                print(f'这是第{i + 1}个数据的Softmax分类结果：\n'
                      f'{softmax_result[i]}\n'
                      f'真实类别是{self.predict_datasets_type[i]},概率最大的为第{self.predict_result[i] + 1}个，代表{self.predict_result[i]}'
                      f'(错)\n')



#创建训练模型实例
nn=NN(x,y) #创建NN实例，其中x代表要分类的数据，y代表每个数据对应的分类
#改变模型参数，设定学习速率为0.01,计算次数为1000次,隐层为100维,输入层输出层维度用np.shape()获得，这样可适用于不同的数据集
imput_num=(np.shape(x)[-1]) #获得输入层的维度，适用于不同的数据集
#获得输出层的维度，即获得数据一共有几类
output_num=0
ls=[]
for i in y:
    if i not in ls:
        output_num+=1
        ls.append(i)
nn.change_Parameters(alpha=0.2,times=1000,input_num=imput_num,hide_num=100,output_num=output_num) #隐层数量可以自定义,此处定义为100层
nn.init_wb()
nn.back_calculate()

#利用模型实例进行预测
np.random.seed(1)
x,y=datasets.make_moons(200, noise=0.20)
nn.predict(predict_datasets=x,predict_datasets_type=y)  #此处还是利用原数据集进行预测

#设置了传入模型参数函数，这样，可以在使用的时候根据不同的学习数据，设定不同的层数维度等参数
#此次的作业，当把代码写好后运行，又出现了一堆报错，主要是因为数组维度的不同而引起的报错；
#然后只有从头开始用笔去运算每个数组的维度，从而发现是从哪一步开始维度开始不同，最后发现自己少写了一个运算，导致这次在此处卡了很久

#导入'dataset_digits'数据集
digits=load_digits()
time1=time.time()
x_train=digits.data[100:300] #生成训练数据集样本,抽取200个样本
y_type=digits.target[100:300] #生成训练数据集类别，为数字0～9
x_predict=digits.data[:100] #生成预测数据集样本
y_predict=digits.target[:100] #生成yuce数据集类别

#训练'dataset_digits'数据集
nn2=NN(datasets=x_train,type=y_type) #创建新的实例
imput_num=(np.shape(x_train)[-1]) #获得输入层的维度，适用于不同的数据集
output_num=0
ls=[]
for i in y_type:
    if i not in ls:
        output_num+=1
        ls.append(i)
nn2.change_Parameters(alpha=0.01,times=1000,input_num=imput_num,hide_num=95,output_num=output_num) #隐层数量可以自定义,此处定义为100层
nn2.init_wb()
nn2.back_calculate()

#预测'dataset_digits'数据集
nn2.predict(predict_datasets=x_predict,predict_datasets_type=y_predict)
print(f'本次训练与预测用时{time.time()-time1}秒')

#显示softmax分类结果
nn2.softmax()

#用Sklearn第三方库进行回归
digits=load_digits()
time1=time.time()
mlp=MLPClassifier()
mlp_x_train=digits.data[100:300] #生成训练数据集样本,抽取200个样本
mlp_y_train=digits.target[100:300] #生成训练数据集类别，为数字0～9
mlp.fit(mlp_x_train,mlp_y_train) #训练
#print(mlp.n_layers_)
mlp_x_predict=digits.data[:100] #生成预测数据集样本
mlp_y_predict=digits.target[:100] #生成yuce数据集类别
mlp_y_result=mlp.predict(mlp_x_predict) #预测
#print(mlp_y_result)
#print(mlp.score(mlp_x_predict,mlp_y_predict))
#可视化
accuracy=0
for i in range(np.shape(mlp_x_predict)[0]):  # 将预测数据集中的每个数据都进行预测
    if mlp_y_result[i]==mlp_y_predict[i]:
        accuracy+=1
        print(f'这是对第{i+1}个数据的预测，真实类别为{mlp_y_predict[i]}，预测类别为{mlp_y_result[i]}')
    else:
        print(f'这是对第{i+1}个数据的预测，真实类别为{mlp_y_predict[i]}，预测类别为{mlp_y_result[i]}           错')
print(f'此次预测个数为{np.shape(mlp_x_predict)[0]}个，正确率为{(accuracy/np.shape(mlp_x_predict)[0])*100}%')
print(f'本次训练与预测用时{time.time()-time1}秒')