import zipfile
import os


# 遍历文件路径下所有文件 打包并发送
def packresult(dir_path):
    """
    压缩指定文件夹
    :param dir_path: 目标文件夹路径
    :return:
    """
    outFullName = dir_path + '.zip'  # 压缩包输出路径
    # baseName1 = os.path.basename(dir_path)
    testcase_zip = zipfile.ZipFile(outFullName, 'w', zipfile.ZIP_DEFLATED)
    for path, dir_names, file_names in os.walk(dir_path):
        # if baseName1 == os.path.basename(path):
        #     continue
        for filename in file_names:
         testcase_zip.write(os.path.join(path, filename), arcname=os.path.basename(path) + '/' + filename)
    testcase_zip.close()
    print("打包成功")

    return outFullName
