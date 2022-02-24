# -*- codeing:uft-8 -*-
import os
import zipfile

dir_path = '/Users/aaron/Hebut/征稿_图像信息安全_20211130截稿/源程序/图塌缩分类/log/drive-download-20220221T024621Z-001/graphSize_200_Normlize_False_20220220_150622_log'
# n = 0
# for path, dir_names, file_names in os.walk(dir_path):
#     n+=1
#     print(path)
#     print(dir_names)
#     for filename in file_names:
#         print(filename)
#     if n > 1:
#         break


def zip_dir(dir_path):
    """
    压缩指定文件夹
    :param dir_path: 目标文件夹路径
    :param outFullName:  压缩文件保存路径+XXXX.zip
    :return:
    """
    outFullName = dir_path + '.zip'
    # baseName1 = os.path.basename(dir_path)
    testcase_zip = zipfile.ZipFile(outFullName, 'w', zipfile.ZIP_DEFLATED)
    for path, dir_names, file_names in os.walk(dir_path):
        # if baseName1 == os.path.basename(path):
        #     continue
        for filename in file_names:
         testcase_zip.write(os.path.join(path, filename), arcname=os.path.basename(path) + '/' + filename)
    testcase_zip.close()
    print("打包成功")

zip_dir(dir_path)



# a = '/Users/aaron/Hebut/征稿_图像信息安全_20211130截稿/源程序/图塌缩分类/log/drive-download-20220221T024621Z-001/graphSize_300_Normlize_True_20220220_112631_log'
# print(a)
# print(os.path.basename(a))
# print(os.path.split(a)[0])