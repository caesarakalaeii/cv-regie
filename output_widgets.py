from abc import ABC
import numpy as np
from logger import Logger
import cv2 as cv
from threading import Thread

class OutputWiget(ABC):
    frame: np.ndarray
    
    
    def update_frame(self, frame):
        self.frame = frame

    
    def start(self):
        raise NotImplementedError()

    
    def stop(self):
        raise NotImplementedError()

        
    def loop(self):
        raise NotImplementedError()
    
class ImageShowWidget(OutputWiget):
    "Wiget to show CameraFeed"
    
    frame: np.ndarray

    def __init__(
        self,
        window_title:str,
        l= Logger()
    ):
        self.l = l
        self.stopped = True
        self.window_title = window_title
        self.frame = None
        self.thread = Thread(target=self.run)
        
   
    def start(self):
        self.stopped = False
        self.thread.start()

    
    def run(self):
        self.l.passingblue("Starting output widget")

        while not self.stopped:
            
            if self.frame is None:
                self.l.info("no frame recieved")
                continue
            
            cv.imshow(self.window_title, self.frame)

            # print(f"Port:{widget.port} - fps:{widget.fps}")

            if cv.waitKey(1) == ord("q"):
                self.stop()
    
    def stop(self):
        self.l.warning('Stopping Output Widget')
        self.stopped = True
        cv.destroyAllWindows()
        
       
  