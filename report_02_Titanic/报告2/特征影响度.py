# 导入第三方库
import pandas as pd
from matplotlib import pyplot as plt


# 读取数据
train_dataset=pd.read_csv('data/train.csv') #导入训练数据集
survived=train_dataset['Survived']
survived_yes=(survived==1).sum() #统计幸存的人数，后续需要使用
#print(survived_yes)
survived_no=(survived==0).sum() #统计未幸存的人数，后续需要使用

# 计算每种特征缺失的数据
blank=train_dataset.isna() #找出缺失的数据
# print(blank)
blank=blank.sum() #统计每一类的缺失数据
print(blank)

# 计算性别与幸存关系(这里将性别进行"编码",male为0,famale为1)
train_dataset.replace('male',int(0),inplace=True) #"编码性别"
train_dataset.replace('female',int(1),inplace=True) #"编码性别"
gender=train_dataset['Sex'] #保存性别组成
print(f'协方差：{gender.cov(survived)}') #计算协方差
print(f'相关系数：{gender.corr(survived)}') #计算相关系数
survived_yes_male=train_dataset[(train_dataset.Sex==0)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
# print(int(survived_yes_male))
survived_no_male=train_dataset[(train_dataset.Sex==0)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_yes_female=train_dataset[(train_dataset.Sex==1)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_no_female=train_dataset[(train_dataset.Sex==1)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
# 绘图
fig=plt.figure(1)
data1=pd.Series({'Survived':int(survived_yes_male),'Unsurvived':int(survived_no_male)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data1.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data1.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='Male', #为饼图添加标题
          )
plt.show()
data2=pd.Series({'Survived':int(survived_yes_female),'Unsurvived':int(survived_no_female)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data2.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data2.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='Female', #为饼图添加标题
          )
plt.show()

# 计算船票等级与幸存关系
Pclass=train_dataset['Pclass'] #保存船票等级组成
print(f'协方差：{Pclass.cov(survived)}') #计算协方差
print(f'相关系数：{Pclass.corr(survived)}') #计算相关系数
survived_yes_firstclass=train_dataset[(train_dataset.Pclass==1)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
# print(int(survived_yes_male))
survived_no_firstclass=train_dataset[(train_dataset.Pclass==1)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_yes_secondclass=train_dataset[(train_dataset.Pclass==2)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_no_secondclass=train_dataset[(train_dataset.Pclass==2)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_yes_thirdclass=train_dataset[(train_dataset.Pclass==3)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_no_thirdclass=train_dataset[(train_dataset.Pclass==3)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
# 绘图
plt.figure(1)
data1=pd.Series({'Survived':int(survived_yes_firstclass),'Unsurvived':int(survived_no_firstclass)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data1.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data1.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='Firstclass', #为饼图添加标题
          )
plt.show()
plt.figure(2)
data2=pd.Series({'Survived':int(survived_yes_secondclass),'Unsurvived':int(survived_no_secondclass)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data2.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data2.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='Secondclass', #为饼图添加标题
          )
plt.show()
plt.figure(3)
data3=pd.Series({'Survived':int(survived_yes_thirdclass),'Unsurvived':int(survived_no_thirdclass)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data3.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data3.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='Thirdclass', #为饼图添加标题
          )
plt.show()

# 同船的兄弟姐妹与幸存关系(有3个及3个以上的样本较少，故只取0,1,2即可)
Sibsp=train_dataset['SibSp'] #保存性别组成
print(f'协方差：{Sibsp.cov(survived)}') #计算协方差
print(f'相关系数：{Sibsp.corr(survived)}') #计算相关系数
survived_yes_noSibsp=train_dataset[(train_dataset.SibSp==0)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
# print(int(survived_yes_male))
survived_no_noSibsp=train_dataset[(train_dataset.SibSp==0)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_yes_oneSibsp=train_dataset[(train_dataset.SibSp==1)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_no_oneSibsp=train_dataset[(train_dataset.SibSp==1)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_yes_twoSibsp=train_dataset[(train_dataset.SibSp==2)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_no_twoSibsp=train_dataset[(train_dataset.SibSp==2)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
# 绘图
plt.figure(1)
data1=pd.Series({'Survived':int(survived_yes_noSibsp),'Unsurvived':int(survived_no_noSibsp)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data1.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data1.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='noSibsp', #为饼图添加标题
          )
plt.show()
plt.figure(2)
data2=pd.Series({'Survived':int(survived_yes_oneSibsp),'Unsurvived':int(survived_no_oneSibsp)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data2.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data2.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='OneSibsp', #为饼图添加标题
          )
plt.show()
plt.figure(3)
data3=pd.Series({'Survived':int(survived_yes_twoSibsp),'Unsurvived':int(survived_no_twoSibsp)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data3.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data3.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='TwoSibsp', #为饼图添加标题
          )
plt.show()

# 同船的父母孩子与幸存关系(有3个及3个以上的样本较少，故只取0,1,2即可)
Parch=train_dataset['Parch'] #保存性别组成
print(f'协方差：{Parch.cov(survived)}') #计算协方差
print(f'相关系数：{Parch.corr(survived)}') #计算相关系数
survived_yes_noParch=train_dataset[(train_dataset.Parch==0)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
# print(int(survived_yes_male))
survived_no_noParch=train_dataset[(train_dataset.Parch==0)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_yes_oneParch=train_dataset[(train_dataset.Parch==1)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_no_oneParch=train_dataset[(train_dataset.Parch==1)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_yes_twoParch=train_dataset[(train_dataset.Parch==2)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_no_twoParch=train_dataset[(train_dataset.Parch==2)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
# 绘图
plt.figure(1)
data1=pd.Series({'Survived':int(survived_yes_noParch),'Unsurvived':int(survived_no_noParch)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data1.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data1.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='noParch', #为饼图添加标题
          )
plt.show()
plt.figure(2)
data2=pd.Series({'Survived':int(survived_yes_oneParch),'Unsurvived':int(survived_no_oneParch)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data2.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data2.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='OneParch', #为饼图添加标题
          )
plt.show()
plt.figure(3)
data3=pd.Series({'Survived':int(survived_yes_twoParch),'Unsurvived':int(survived_no_twoParch)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data3.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data3.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='TwoParch', #为饼图添加标题
          )
plt.show()

# 登船港口幸存关系(这里我们需要对港口名字进行"编码"，S为0,C为1,Q为2)
train_dataset.replace('S',int(0),inplace=True) #"编码"港口
train_dataset.replace('C',int(1),inplace=True) #"编码"港口
train_dataset.replace('Q',int(2),inplace=True) #"编码"港口
Embarked=train_dataset['Embarked'] #保存性别组成
print(f'协方差：{Embarked.cov(survived)}') #计算协方差
print(f'相关系数：{Embarked.corr(survived)}') #计算相关系数
survived_yes_EmbarkedS=train_dataset[(train_dataset.Embarked==0)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
# print(int(survived_yes_male))
survived_no_EmbarkedS=train_dataset[(train_dataset.Embarked==0)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_yes_EmbarkedC=train_dataset[(train_dataset.Embarked==1)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_no_EmbarkedC=train_dataset[(train_dataset.Embarked==1)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_yes_EmbarkedQ=train_dataset[(train_dataset.Embarked==2)&(train_dataset.Survived==1)]['Survived'].value_counts() #筛选两个条件都满足的样本
survived_no_EmbarkedQ=train_dataset[(train_dataset.Embarked==2)&(train_dataset.Survived==0)]['Survived'].value_counts() #筛选两个条件都满足的样本
# 绘图
plt.figure(1)
data1=pd.Series({'Survived':int(survived_yes_EmbarkedS),'Unsurvived':int(survived_no_EmbarkedS)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data1.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data1.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='Embarked On S', #为饼图添加标题
          )
plt.show()
plt.figure(2)
data2=pd.Series({'Survived':int(survived_yes_EmbarkedC),'Unsurvived':int(survived_no_EmbarkedC)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data2.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data2.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='Embarked On C', #为饼图添加标题
          )
plt.show()
plt.figure(3)
data3=pd.Series({'Survived':int(survived_yes_EmbarkedQ),'Unsurvived':int(survived_no_EmbarkedQ)}) #创建绘图序列组成
plt.axes(aspect='equal') #控制饼图为正圆
data3.name=' ' #将序列的名称设置为空字符，否则绘制的饼图左边会出现None这样的字眼
data3.plot(kind='pie', #饼状
           autopct='%.1f%%', #饼图中添加数值标签
           title='Embarked On Q', #为饼图添加标题
          )
plt.show()

# 计算船票价格与幸存
# train_dataset['Fare'].hist()
train_dataset[train_dataset.Fare<=100]['Fare'].hist() #查看船票价格分布
Fare=train_dataset['Fare']
print(f'相关系数：{Fare.corr(survived)}') #计算相关系数






# 一开始统计缺失的数据时，统计成表格中有多少个0了，就发现结果里'Survived'有很多个，就感觉不太对，然后才发现应该用统计缺项的函数，而不是0
#pandas统计两个判据，画图(画子图与subplot不一样，series）