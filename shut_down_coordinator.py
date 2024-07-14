

from logger import Logger


class Shutdown_Coordinator:
    '''
    Class to coordinate shutdowns, only use stop() and running()
    '''
    
    __running: bool
    
    def __init__(self, l = Logger(True)) -> None:
        self.__running = True
        self.l = l
    
    def stop(self):
        if self.__running:
            self.l.fail('Stop called, coordinating shutdown')
            self.__running = False
    
    def running(self):
        return self.__running