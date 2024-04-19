import pickle


class SerializationUtils:
    @staticmethod
    def serialize(obj, file_path):
        """将对象序列化到文件。

        参数:
        obj -- 要序列化的对象
        file_path -- 目标文件路径
        """
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def deserialize(file_path):
        """从文件反序列化对象。

        参数:
        file_path -- 源文件路径

        返回:
        反序列化的对象
        """
        with open(file_path, 'rb') as file:
            return pickle.load(file)
