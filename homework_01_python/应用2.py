import os
def findfiles(path,all,paths,filetype):
    filelist = os.listdir(path) #列出所有的文件及文件夹
    # 判断每个filelist里是否为文件夹，若为文件夹则进行递归处理
    for file in filelist:
        name= os.path.join(path,file)   #获得文件或文件夹全名
        if os.path.isdir(name):         # 判断是否是文件夹并进行递归
            findfiles(name,all,paths,filetype)
        elif file.endswith(filetype):   #如果是文件，则判断文件类型是否与寻找的文件类型一致
            all.append(file)    #添加符合类型的文件
            paths.append(path)  #同时记录该文件的路径
    return all,paths

if __name__=='__main__':
    searchpath=input('请输入搜索路径：')
    filetype=input('请输入搜索文件类型(如py或md)：')
    files,paths=findfiles(searchpath, [],[],filetype)
    if len(files)!=0:
        for f in range(0,len(files)):
            print(f'文件名：{files[f]}     路径：{paths[f]}')  #输出符合类型的文件以及文件的路径
        print(f'共搜寻到文件{len(files)}个')
    else:print('提示：未搜索到该类型文件！')