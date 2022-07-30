# 导入第三方库
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 处理掉不需要的特征
test_dataset=pd.read_csv('data/test.csv') #导入训练数据集
test_dataset=test_dataset.drop(['PassengerId','Name','Ticket'],axis=1,) #处理不需要的特征列

# 找到缺失的量
blank=test_dataset.isna() #找出缺失的数据
# print(blank)
blank=blank.sum() #统计每一类的缺失数据
print(blank)

# 处理Fare
Fare_null_dataset=test_dataset[test_dataset['Fare'].isnull()] #找到那个样本
# print(Fare_null_dataset)
Fare_dataset=test_dataset[test_dataset['Pclass']==3]
Fare_average=Fare_dataset['Fare'].mean()
test_dataset['Fare']=test_dataset['Fare'].fillna(Fare_average) #将平均值补齐
print(test_dataset['Fare'])
print(f'Fare平均值为{Fare_average}元')

# Cabin
test_dataset=test_dataset.drop(['Cabin'],axis=1)



# 预测
test_dataset=pd.read_csv('data/test.csv') #导入训练数据集
test_dataset.drop(['PassengerId','Name','Ticket'],axis=1,) #处理不需要的特征列
test_dataset.replace('male',int(0),inplace=True) #"编码性别"
test_dataset.replace('female',int(1),inplace=True) #"编码性别"
age_dataset=test_dataset[['Pclass','Sex','SibSp','Parch','Fare','Age']] #将用于"年龄预测模型"的样本放入
# print(age_dataset)
age_test_dataset=age_dataset[age_dataset['Age'].notnull()]#将有年龄的样本当作训练集
# print(age_train_dataset)
age_predict_dataset=age_dataset[age_dataset['Age'].isnull()]#将没有年龄的样本作为预测集
age_test_dataset_labels=age_test_dataset['Age'] #将标签拿出来
age_test_dataset=age_test_dataset.drop(['Age'],axis=1) #将标签从训练集扔掉
age_predict_dataset=age_predict_dataset.drop(['Age'],axis=1) #将标签从预测集扔掉
# 训练及预测
rfc=RandomForestClassifier() #创建模型实例
rfc.fit(age_test_dataset,age_test_dataset_labels.astype('int')) #训练
age_predict_dataset_labels=rfc.predict(age_predict_dataset) #预测
print(f'预测的年龄为{age_predict_dataset_labels}')
# 填充
test_dataset.loc[test_dataset['Age'].isnull(),'Age']=age_predict_dataset_labels

# embarked
Embarked_encode=pd.get_dummies(test_dataset['Embarked'],prefix='Embarked') #进行"独热编码"
print(Embarked_encode)
test_dataset=pd.concat([test_dataset,Embarked_encode],axis=1) #将新编码融入训练集中
test_dataset=test_dataset.drop(['Embarked'],axis=1) #将老的特征列丢掉

# Pclass
Pclass_encode=pd.get_dummies(test_dataset['Pclass'],prefix='Pclass') #进行"独热编码"
print(Pclass_encode)
test_dataset=pd.concat([test_dataset,Pclass_encode],axis=1) #将新编码融入训练集中
test_dataset=test_dataset.drop(['Pclass'],axis=1) #将老的特征列丢掉

