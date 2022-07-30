import string
import 应用2 # 用于搜寻所有py或cpp文件

def count_lines(file):
    #打开文件并初始化参数
    with open(file,'r', encoding='utf8') as f:
        lines=f.readlines()
        lines_zhushi=0
        lines_code=0
        n=0
        f.seek(0)
    #统计总代码行数量，因为readlines不会读取空格行，所以需用readline读取
        while True:
            total=f.readline()
            if total:
                n += 1
            else:
                break
    #开始统计每种类型数量
        for i in range(0,len(lines)):
            ls=lines[i]
            for j in range(0,len(ls)):  #判断第j行类型
                if ls[j] == '#':
                    lines_zhushi += 1
                    break
                if ls[j] in string.digits+string.ascii_letters:
                    lines_code +=1
                    break
        return_lines = [lines_code, lines_zhushi, n-lines_code-lines_zhushi]  # 返回计算值
    return return_lines

if __name__=='__main__':
    searchpath = input('请输入搜索路径：')
    filetype = input('请输入搜索文件类型(如py或cpp)：')
    files, paths = 应用2.findfiles(searchpath, [], [], filetype)  #利用应用2的文件类型匹配函数
    total_code,total_zhushi,total_block=0,0,0
    lines = [total_code, total_zhushi, total_block]
    if len(files) != 0:
        for f in range(0, len(files)):
            return_lines=count_lines(files[f])
            lines[0]=lines[0]+return_lines[0]
            lines[1]=lines[1]+return_lines[1]
            lines[2]=lines[2]+return_lines[2]
            print(f'文件名：{files[f]}     路径：{paths[f]}')
            print(f'共有代码行：{return_lines[0]},共有注释行:{return_lines[1]},共有空格行:{return_lines[2]}')
            print()
        print(f'总共搜索到{len(files)}个{filetype}文件,共计代码行：{lines[0]},共计注释行：{lines[1]}，共计空格行：{lines[2]}')
    else:print('提示：未搜索到该类型文件！')

#功能：输入一个地址，以及统计文件的类型（如py或cpp）。程序会自动搜索该目录下的对应类型文件，并对其各种行数进行统计、输出。
#如果注释写在代码后面，则不计入注释行数