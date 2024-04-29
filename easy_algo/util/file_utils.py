import os


def get_all_files(directory, suffix='txt'):
    # 获取目录下所有的文件和文件夹
    files_and_folders = os.listdir(directory)

    # 筛选出所有的.txt文件
    txt_files = [os.path.join(directory, file) for file in files_and_folders if suffix is None or file.endswith(suffix)]
    return txt_files
