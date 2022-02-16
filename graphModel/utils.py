# 存放需要的小功能函数
import os


def ensure_dir(file_path):
    '''
    :param file_path:  创建文件夹
    :return: 检查 并 创建
    '''
    try:
        os.mkdir(file_path)
    except FileExistsError:
        print("Folder already found")

# if __name__ == '__main__':
#     ensure_dir('1')