import requests
import re
import os
import time
import random
from shutil import copy2
import cv2

def run_getdata():
    # 伪装header，防止识别为爬虫
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,like Gecko) Chrome/84.0.4147.125 Safari/537.36'}

    ls=['T恤','凉鞋','包','套头衫','短靴','外套','衬衫','裤子','运动鞋','连衣裙']
    for i in range(len(ls)):
        # name=input('请输入爬取的图片类别：')
        # 初始化后续使用变量
        num = 0
        num_again = 0  # 用于保存当前路径(后续使用)
        num_problem = 0  # 也用于保存文件路径
        # 得到输入图片数量变量(60根据搜索的url所定)
        name=ls[i]
        x=10
        # x=input('请输入要爬取的图片数量？ 1等于60张图片，以此类推：')
        list_1=[]
        filename_total=[]
        file_name=ls[i]
        # 计时
        time_start=time.time()
        for i in range(int(x)):
            # os.getcwd()获得当前的路径
            name_1=os.getcwd()
            # os.path.join连接两个或更多的路径名组件(这里用于保存图片储存路径，保存在当前路径中的/data文件夹)
            name_2=os.path.join(name_1,'data/'+file_name)
            # 创建爬取网站，这里我默认使用Baidu
            url='https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='
            url=url+name+'&pn='+str(i*30)

            # # 调用request函数，构造一个向服务器请求资源的url对象，返回从服务器获取的所有的相关资源
            # res=requests.get(url,headers=headers)
            # htlm_1=res.content.decode()

            # 这里使用我打算不实用上述的经典get，打算使用Session进行创建对象，之前在网上看到过使用Session可以服务器保持连接，使爬取速度加快
            res=requests.Session()
            htlm_1=res.get(url,headers=headers).content.decode()

            # 正则筛选
            # 匹配查找URL
            a=re.findall('"objURL":"(.*?)",',htlm_1)
            # 检测是否存在路径(这里用于保存图片储存路径，name_2表示保存在当前路径中的/data文件夹)
            if not os.path.exists(name_2):
                # 创建多层目录
                os.makedirs(name_2)
            for b in a:
                try:
                    # 添加获取图片的对应网址(为了保证爬取的图片不重复(至少不是相同的网址))
                    b_1=re.findall('https:(.*?)&',b)
                    b_2=''.join(b_1)
                    # 如果没有爬取过这个图片，就进行储存
                    if b_2 not in list_1:
                        num=num+1 #用于图片计数
                        # 创建Session
                        imgget=requests.Session()
                        img=imgget.get(b) #获取爬取图片
                        # 新建图片文件进行保存，图片名为XX第几张
                        filename=os.path.join(name_1,'data/'+file_name,name+str(num)+'.png')
                        f=open(filename,'ab')
                        # 打印正在保存第几张
                        print(f'正在下载第{str(num)}张图片')
                        f.write(img.content)
                        f.close()
                        # 将这个图片的信息保存在列表里，保证不重复下载图片
                        list_1.append(b_2)
                        # 保存每个图片的文件名
                        filename_total.append(filename)
                    elif b_2 in list_1:
                        num_again=num_again+1
                        continue
                except Exception :
                    print(f'第{str(num)}张图片下载出现问题')
                    num_problem=num_problem+1
                    continue

        # 停止计时
        time_end=time.time()
        print(f'爬取图片完成，总共爬取{num+num_again+num_problem}张,已保存{num}张,重复爬取{num_again}张,爬取出现问题{num_problem}张')
        print(f'总计爬取用时{time_end-time_start}秒')
        # 检测空文件
        # 创建统计空文件数量变量
        num_blank=0
        # 遍历所有刚才下载的文件呢
        for i in filename_total:
            # print(i)
            # 如果文件大小小于200B，则当作空图片(个人感觉图片应该不止200B)
            if os.path.getsize(i)<200:
                print(f'{i}为空图片')
                # 删除该文件
                os.unlink(i)
                num_blank+=1
                continue
        if num_blank==0:
            print('此次下载没有空图片')
        else:
            print(f'经检测，下载到共有{num_blank}张空图片，已删除')

run_getdata()

