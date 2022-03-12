import time
import os
import re
#
# timestr = 'Fri Feb 25 18:23:08 CST 2022'
#
#
# time1 = time.strptime(timestr, '%a %b %d %H:%M:%S CST %Y')
# # print(time.mktime(time1))
# # print(time.time())
# # timestr2 = time.strftime('%Y%m%d_%H%M%S', time.strptime(timestr, '%a %b %d %H:%M:%S CST %Y'))
# # print(f'time1: {time1}')
# # print(f'localtime: {time.localtime()}')
# # print(abs(time.time() - time.mktime(time1)))
# sys_cst_time = os.popen('date').read().strip('\n')
#
# print(sys_cst_time)
# print('##')
# if abs(time.mktime(time.strptime(str(sys_cst_time), '%a %b %d %H:%M:%S CST %Y')) - time.time()) < 120:
#     print('ok')
# else:
#     print(abs(time.mktime(time.strptime(timestr, '%a %b %d %H:%M:%S CST %Y')) - time.time()))

# a = '/Users/aaron/Hebut/征稿_图像信息安全_20211130截稿/源程序/图塌缩分类/DynamicCANGraphLen/utils'
# print(os.path.split(a)[0])
# print(os.path.basename(a))

# a = 'epoch_01_20220225_002202critic_2_target.pth'
# print(re.search(r"\d+", a).group(0))
# print(os.path.splitext(a)[1])

# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
# handler = logging.FileHandler("log.txt")
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
#
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
#
# logger.addHandler(handler)
# logger.addHandler(console)
#
# logger.info("Start print log")
# logger.debug("Do something")
# logger.warning("Something maybe fail.")
# logger.info("Finish")

# # 整数
# print("显示占位符%d---"%(1314))#%d表示占位整数
# print("显示占位符%3d--"%(1314))#%3d表示占位整数不小于3个位数
# print("显示占位符%13d--"%(1314))#%13d表示占位整数不小于13个位数，不满足在整数左边空格补齐
# print("显示占位符%-13d--"%(1314))#%13d表示占位整数不小于13个位数，不满足在整数右边空格补齐
# print("显示占位符%07d--"%(1314))#%07d表示占位整数不小于7个位数，不满足在整数左边0补齐;不存在右边0补齐的情况，整数右边补0，整数的数值发生变化
#
# print('================================')
# # 浮点数
# print("浮点数%f-"%(3.14))# %f补齐六位小数点
# print("浮点数%g-"%(3.14))# %g显示小数，末位的0不显示
# print("浮点数%g-"%(3.14000))

#%m.nf n 表示保留几个小数，m表示占用的列数；小数点也算一位，保留小数时会进行四舍五入，默认补齐位数空格左补齐；使用“-”表示右补齐；优先执行小数点位数的显示，然后计算补齐位数，一般m>n+2
# print("浮点数%7.3f--"%(3.14))
# print("浮点数%-7.3f--"%(3.14))
# print("浮点数%2.5f--"%(3.14))
# print("浮点数%3.2f--"%(3.14))
#
# print(f'{3.14:<10.3g}')

# mylist = []
# x = [1, 2, 3]
#
# a = 1
# mylist.append(a)
# x[0] = 3
# a = 2
# mylist.append(a)
# print(mylist)
# from pandas import Series, DataFrame
# import pandas as pd
#
# data = {'水果': Series(['苹果', '梨', '草莓', '葡萄', '香蕉']),
#         '数量': Series([3, 2, 5, 7, 9]),
#         '价格': Series([10, 9, 8, 4, 2])}
# df = DataFrame(data)
# df_train = df.iloc[:20, :]
# # df_val = df.iloc[2:, :]
# print(df)
# print(df_train)
# print(df_val)

print('%.4f' % 3.1415926)