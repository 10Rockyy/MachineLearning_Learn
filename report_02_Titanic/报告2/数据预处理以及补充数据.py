# 导入第三方库
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# 处理掉不需要的特征
train_dataset=pd.read_csv('data/train.csv') #导入训练数据集
train_dataset=train_dataset.drop(['PassengerId','Name','Ticket'],axis=1,) #处理不需要的特征列

# 补充Age数据
# 平均值补全
age_average=train_dataset['Age'].mean() #计算年龄平均值
print(f'年龄平均值为{age_average}岁')
train_dataset['Age']=train_dataset['Age'].fillna(age_average) #将平均值补齐
# print(train_dataset['Age'])

# 预测
train_dataset=pd.read_csv('data/train.csv.csv') #导入训练数据集
train_dataset.drop(['PassengerId','Name','Ticket'],axis=1,) #处理不需要的特征列
train_dataset.replace('male',int(0),inplace=True) #"编码性别"
train_dataset.replace('female',int(1),inplace=True) #"编码性别"
age_dataset=train_dataset[['Pclass','Sex','SibSp','Parch','Fare','Age']] #将用于"年龄预测模型"的样本放入
# print(age_dataset)
age_train_dataset=age_dataset[age_dataset['Age'].notnull()]#将有年龄的样本当作训练集
# print(age_train_dataset)
age_predict_dataset=age_dataset[age_dataset['Age'].isnull()]#将没有年龄的样本作为预测集
age_train_dataset_labels=age_train_dataset['Age'] #将标签拿出来
age_test_dataset=age_train_dataset.drop(['Age'],axis=1) #将标签从训练集扔掉
age_predict_dataset=age_predict_dataset.drop(['Age'],axis=1) #将标签从预测集扔掉
# 训练及预测
rfc=RandomForestClassifier() #创建模型实例
rfc.fit(age_train_dataset,age_train_dataset_labels.astype('int')) #训练
age_predict_dataset_labels=rfc.predict(age_predict_dataset) #预测
print(f'预测的年龄为{age_predict_dataset_labels}')
# 填充
train_dataset.loc[train_dataset['Age'].isnull(),'Age']=age_predict_dataset_labels

# 删除Cabin
train_dataset=train_dataset.drop(['Cabin'],axis=1)

# 处理Embarked
Embarked_null_dataset=train_dataset[train_dataset['Embarked'].isnull()]
print(Embarked_null_dataset)
Embarked_Ticket_dataset=train_dataset[train_dataset['Ticket']==str(113572)]
print(Embarked_Ticket_dataset)
train_dataset['Embarked']=train_dataset['Embarked'].fillna("S") #补齐

print(train_dataset)
train_dataset=train_dataset.drop(['PassengerId','Name','Ticket'],axis=1) #扔掉

# embarked编码
Embarked_encode=pd.get_dummies(train_dataset['Embarked'],prefix='Embarked') #进行"独热编码"
# print(Embarked_encode)
train_dataset=pd.concat([train_dataset,Embarked_encode],axis=1) #将新编码融入训练集中
train_dataset=train_dataset.drop(['Embarked'],axis=1) #将老的特征列丢掉

# Pclass
Pclass_encode=pd.get_dummies(train_dataset['Pclass'],prefix='Pclass') #进行"独热编码"
print(Pclass_encode)
train_dataset=pd.concat([train_dataset,Pclass_encode],axis=1) #将新编码融入训练集中
train_dataset=train_dataset.drop(['Pclass'],axis=1) #将老的特征列丢掉



# 数据处理花了很长时间，数据对于机器学习很重要，甚至补充中都用到了机器学习
# 对于pandas的Series和DataFrame和画图更加熟悉
# notebook是dbug好帮手
# Cabin不能删
# 回头看第一个报告是真的太简单了