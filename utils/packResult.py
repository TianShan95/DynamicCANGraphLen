import zipfile
import os
import re


# 遍历文件路径下所有文件 打包并发送
def packresult(dir_path, epoch):
    """
    压缩指定文件夹
    :param epoch: 训练到的 代数
    :param dir_path: 目标文件夹路径 不带最后的斜杠 /
    :return:
    """

    outFullName = dir_path + '.zip'  # 压缩包输出路径
    print(f'压缩文件 {outFullName}')
    # baseName1 = os.path.basename(dir_path)
    testcase_zip = zipfile.ZipFile(outFullName, 'w', zipfile.ZIP_DEFLATED)
    print(f'遍历文件夹 {dir_path}')
    for path, dir_names, file_names in os.walk(dir_path):
        # if baseName1 == os.path.basename(path):
        #     continue
        print(f'进入文件夹 {path}')
        for filename in file_names:
            # 打包 1.指定 epoch 的模型 2. log文件
            # 打包最近两代的模型文件 以防最新的一代没保存好
            if int(re.search(r"\d+", filename).group()) == epoch or int(re.search(r"\d+", filename).group()) == epoch-1 or os.path.splitext(filename)[1] == '.log' :
                print(f'打包 {filename}')
                testcase_zip.write(os.path.join(path, filename), arcname=os.path.basename(path) + '/' + filename)
    testcase_zip.close()
    print("打包成功")
    print(f'打包文件输出路径 {outFullName}')
    return outFullName

if __name__ == '__main__':
    a = 'a'
    packresult(a)