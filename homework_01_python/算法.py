# 使用冒泡法排序
# a=input('请输入数列(数字间用,隔开):')
# ls=a.split(',')
# ls=list(map(float,ls))   #将列表里面的str类型转换为float型
# n= len(ls)        #计算列表长度
# for i in range(0, n):
#         for j in range(0, n-i-1):
#             if ls[j]>ls[j+1]:
#                 ls[j],ls[j+1]=ls[j+1],ls[j]
# ls=ls[::-1]     #列表倒叙排列
# print(ls)

# 使用sorted函数
a=input('请输入数列(数字间用,隔开):')
ls=a.split(',')
ls=list(map(float,ls))
ls=sorted(ls,reverse=True)
print(ls)