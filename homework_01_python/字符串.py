passage='One       is always on a strange road, watching strange scenery and listening to strange music. Then one day, you will find that the things you try hard to forget are already gone. '
con=passage.lower().replace(',',' ').replace('.',' ').replace('\\',' ').replace('/',' ').replace('?',' ').split()
#对文章所有单词改为小写，将所有标点替代为空格
dic={}
for i in con:
    dic[i]=dic.get(i,0)+1
dic=sorted(dic.items(),key=lambda item:item[1],reverse=True)
print(dic)

# 思考：
# 1.有两个空格不会影响统计，split()函数会默认两个空格都为分隔符。
#   有\t会影响统计，会当成转义字符进行处理，可用replace()函数将\替代为空格
# 2.有标点以及/，可以用replace()函数进行替换为空格