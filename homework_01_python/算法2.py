def findtarget(matrix,target):
    m=len(matrix[0])
    for i in range(0,len(matrix)):
        for j in range(0,m):
            if target ==matrix[i][j]:
                return True
            elif target <matrix[i][j]:
                m=j
    return False

if __name__=='__main__':
    matrix=[[1, 4, 7, 11, 15],
    [2, 5, 8, 12, 19],
    [3, 6, 9, 16, 22],
    [10, 13, 14, 17, 24],
    [18, 21, 23, 26, 30]]
    target=int(input('请输入目标数值：'))
    print(findtarget(matrix,target))

#方法解释：先对第一行从第一列到最后一列进行比较，如果发现target小于第一行第m列的数字，则之后所有行的第m及更后的列的数字就不用比较了。
#        同理，对第二行、第三行进行相同的处理，就可以节省比较数字的时间。