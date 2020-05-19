
from multiprocessing import Manager

class Counter(object):
    def __init__(self, manager: Manager, value: int):
        self.__value = manager.Value('i', value)
        self.__lock = manager.Lock()

    def increment(self):
        with self.__lock:
            self.__value.value += 1

    @property
    def lock(self):
        return self.__lock

    @property
    def value(self):
        return self.__value.value