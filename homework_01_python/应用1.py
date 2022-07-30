import string
import random
import json



q=string.digits+string.ascii_letters #生成数字与字母，其中字母区分大小写
l=int(input('请输入优惠卷代码长度：'))
coupons=[]
for i in range(1,201):
    coupon = ''
    for j in range(1,l+1):
        coupon=random.choice(q)+coupon #根据输入优惠码位数进行字符串相加
    if coupon not in coupons:    #判断是否有重复的优惠码
        coupons.append(coupon)
for k in range(0,len(coupons)):
    print(coupons[k],f'第{k+1}张优惠券')

with open('生成的优惠券.json','a') as f:
    json.dump(coupons,f)



