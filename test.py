import time

timestr = 'Fri Feb 25 18:23:08 CST 2022'


time1 = time.strptime(timestr, '%a %b %d %H:%M:%S CST %Y')
# print(time.mktime(time1))
# print(time.time())
# timestr2 = time.strftime('%Y%m%d_%H%M%S', time.strptime(timestr, '%a %b %d %H:%M:%S CST %Y'))
print(f'time1: {time1}')
print(f'localtime: {time.localtime()}')
print(time.time() - time.mktime(time1))
