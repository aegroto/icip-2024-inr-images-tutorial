from typing import Self


class ByteStream:
    def __init__(self, data=None):
        if data is None:
            data = bytes()

        self.__data = data

    def append(self, other: Self):
        self.write(other.get_bytes())

    def get_bytes(self):
        return self.__data

    def write(self, new_data: bytes):
        self.__data += new_data

    def read(self, length: int) -> bytes:
        data_to_read = self.__data[:length]
        self.__data = self.__data[length:]
        return data_to_read

    def __len__(self):
        return len(self.__data)
