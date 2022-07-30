#导入第三方库
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

#导入加载数据
digits = load_digits()

def visualize(dataset,i):
    #cv.imshow('img',dataset.images[i])
    #cv.waitKey()
    plt.imshow(dataset.images[i],cmap=plt.cm.binary) #显示该错误分类的图像
    plt.show()

#预处理(准备学习与预测数据)
#此处为了对比我敲打的代码，我选择了与第一问同样的学习数据、并同时选取了最后100个数据进行预测，看Sklearn需要用多长时间
num=int((1/2)*(np.shape(digits.data)[0])) #抽取一半的数据进行学习，与第一问相同
train_data=digits.data[:num] #将学习需用的数据储存
train_data_target=digits.target[:num] #将学习的数据标签储存
num_predict=100 #预测数据数量，与第一问相同
predict_data=digits.data[-1-num_predict:-1] #将需要预测数据储存
predict_data_target=digits.target[-1-num_predict:-1] #将预测数据标签储存，以对比正确度

#调用Sklearn训练
logic=LogisticRegression()
logic.fit(train_data, train_data_target) #训练
predict_result=logic.predict(predict_data) #预测
predict_accuracy=accuracy_score(predict_data_target,predict_result) #预测准确率
accuracy,wrong=0,0
for i in range(len(predict_data)):
    if predict_data_target[i] == predict_result[i]:
        print(f'第{i+1}次预测,真值为{predict_data_target[i]},预测值为{predict_result[i]}')
        accuracy += 1
    else:
        print(f'第{i+1}次预测,真值为{predict_data_target[i]},预测值为{predict_result[i]}    错')
        visualize(digits,i)
        wrong += 1
#print(predict_accuracy)
print(f'本次多元预测模型,共预测{num_predict}个样本，总计正确{accuracy}个，错误{wrong}个，正确率为{(predict_accuracy*100)}%')

#程序在引用第三方库的时候报错，在网上查询了，解释该报错：
#迭代总数达到限制。增加迭代次数（最大值）或缩放数据.scikit-learn版本问题导致的匹配警告和收敛警告，调用的函数和方法已经有改变，暂时不处理也不会影响程序运行。建议升级版本或者修改为最新版本的用法，合理使用库函数！





