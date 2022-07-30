l=0
for j in range(1,10):
    for k in range(1,10):
        if(j>=k):
            l=j*k
            print(f'{j}*{k}={l}',end=' ')
            if(j==k):
                print()