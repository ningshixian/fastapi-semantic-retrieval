import json
import oss2
import sys
import os


def list_all_file(path, all_files=None):
    if all_files is None:
        all_files = []
    if os.path.exists(path):
        files = os.listdir(path)
    else:
        raise FileExistsError("{} is not exist!".format(path))

    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            list_all_file(os.path.join(path, file), all_files)
        else:
            all_files.append(os.path.join(path, file))
    return all_files


class OssService(object):
    """对象存储类，将模型传至阿里云端"""

    def __init__(self, access_key_id, access_key_secret, endpoint, bucket_name):
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(self.auth, endpoint, bucket_name, connect_timeout=5)  # 连接OSS

    def put_file(self, file_path, oss_path):
        """上传文件"""
        with open("{}".format(file_path), "rb") as f:
            put_result = self.bucket.put_object(oss_path, f)
        if put_result.status == 200:
            print("put success")

    def get_file(self, file_path, oss_path):
        # oss_path: oss上bucket中的文件名
        # file_path: 保存在当地的文件路径+文件名
        def percentage(consumed_bytes, total_bytes):
            if total_bytes:
                rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
                print('\r{0}% '.format(rate), end='')
                sys.stdout.flush()

        get_result = self.bucket.get_object_to_file(oss_path, file_path, progress_callback=percentage)
        if get_result.status == 200:
            print("get success")
        else:
            print("get failed")

    def put_folder_to_path(self, folder_path: str, oss_path: str):
        files = list_all_file(folder_path)
        prefix = oss_path.strip("/")
        for file in files:
            file_name = os.path.relpath(file, folder_path)
            self.put_file(file, "/".join([prefix, file_name]))

    def get_folder_to_path(pretrain_file, oss_get_path):
        for obj in oss2.ObjectIterator(self.bucket, prefix = oss_get_path, delimiter = '/'):
            # 通过is_prefix方法判断obj是否为文件夹。
            if obj.is_prefix():  # 判断obj为文件夹。
                print('directory: ' + obj.key)
            else:                # 判断obj为文件。
                print('file: ' + obj.key)
                file_name = str(obj.key).split('/')[-1]
                if file_name:
                    self.get_file(pretrain_file+file_name, obj.key)


# # oss配置
# oss_server = OssService(
#     access_key_id=...,
#     access_key_secret=...,
#     endpoint=...,
#     bucket_name=...,
# )

# print("通过OSS上传数据....")
# oss_server.put_file(..., ...)

# print("通过oss拉取数据")
# oss_server.get_file(..., ...)

