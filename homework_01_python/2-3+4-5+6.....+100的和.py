#方法1：
m=0
for i in range(2,101):
    m=((-1)**i)*i+m
print(f'方法1结果：{m}')

#方法2：
a,b = 2,0            
while a <= 100:
    if a % 2 == 0:
        b=a+b
    else:
        b=b-a
    a+=1
print(f'方法2结果：{b}')

#方法3：
q=sum(range(2,101,2))-sum(range(3,100,2))  #正数与负数先组合，再相加减即可
print(f'方法3结果：{q}')