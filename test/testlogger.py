# 对比 logger 和 f.write 的写入文件速度
# 写 10000 行 "文件写入速度测试"
import time
import logging
import os




begin = time.time()
# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FORMAT = "%(asctime)s - %(message)s"
# LOG_FORMAT = "%(message)s"

for i in range(10000):
    logging.basicConfig(filename='fileWriteTest1.txt', level=logging.DEBUG, format=LOG_FORMAT)

    logging.info("文件写入速度测试")
print(f'logger 用时 {time.time() - begin}')


begin1 = time.time()
for i in range(10000):
    with open('fileWriteTest2.txt', 'a') as f:
        f.write("文件写入速度测试\n")
        f.close()
print(f'f.write 用时 {time.time() - begin1}')


time.sleep(1)
os.remove('fileWriteTest1.txt')
os.remove('fileWriteTest2.txt')