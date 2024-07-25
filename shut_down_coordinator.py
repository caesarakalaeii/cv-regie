from multiprocessing import Value
from ctypes import c_bool
from logger import Logger


class Shutdown_Coordinator:
    '''
    Class to coordinate shutdowns, only use stop() and running()
    '''

    def __init__(self, l=Logger(True)) -> None:
        self.__running = Value(c_bool, True)  # multiprocessing safe boolean
        self.l = l

    def stop(self):
        with self.__running.get_lock():  # Acquire lock to ensure thread-safe changes
            if self.__running.value:
                self.l.fail('Stop called, coordinating shutdown')
                self.__running.value = False

    def running(self):
        return self.__running.value
