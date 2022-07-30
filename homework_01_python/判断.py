l1=[100,60,40,20,10,0]
l2=[0.01,0.015,0.03,0.05,0.075,0.1]
a=float(input('当月利润为:'))
j,k=0,0
for i in range(0,6):
    if a >=l1[i]:
        k=a-l1[i]
        j=k*l2[i]+j
        a=l1[i]
print(f'奖金为为{j}万元')
