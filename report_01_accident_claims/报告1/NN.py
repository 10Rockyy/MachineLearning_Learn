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




#创建Sigmoid函数
def sigmoid(x):
    return np.longfloat(1.0/(1+np.exp(-x)))

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
            # print(f'这是第{i+1}次训练,正确率为{self.accuracy*100}%')
            if i%50==0:
                print(f'经过{i}次训练，模型预计精准度为{self.accuracy*100}%')
            elif i==self.times:
                print(f'经过{i}次训练，模型预计精准度为{self.accuracy*100}%')
            if self.accuracy==1:
                print(f'经过{i}次训练，模型预计精准度为100%')
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

    def predict(self,predict_datasets,predict_datasets_type=[0]):
        if all(predict_datasets_type)!=[0]:
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
            return predict_result
        else:
            return
    def predict_no_orgin_labels(self,predict_datasets,predict_datasets_type=[0]):
        self.datasets = predict_datasets
        self.forward_calculate()  # 利用训练好的w,b进行向前运算
        predict_result = np.argmax(self.z2, axis=1)
        return predict_result

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

