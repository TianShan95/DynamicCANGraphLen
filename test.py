# import time
import os
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

a = '/Users/aaron/Hebut/征稿_图像信息安全_20211130截稿/源程序/图塌缩分类/DynamicCANGraphLen/utils'
print(os.path.split(a)[0])
print(os.path.basename(a))