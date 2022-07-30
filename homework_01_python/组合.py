m=0
for i in  range(1,5):
    for j in range(1,5):
        if(i!=j):
            for k in range(1,5):
                if(j!=k and k!=i):
                    m=m+1
                    print(f"{i}{j}{k}",end=' ')
print()
print(f'一共有{m}种组合')

#思考：一共用到三次循环,通过在执行第三个循环之前先判断第一二位是否相等，可以减少第三次循环次数