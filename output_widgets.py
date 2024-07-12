from abc import ABC
from detection_widgets import DetectionWidget
from utilities import calculate_frame_box_static
import numpy as np
from logger import Logger
import cv2 as cv
from threading import Thread

class OutputWiget(ABC):
    
    def update_frame(self, frame):
        raise NotImplementedError()
    
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

        while not self.stopped:
            
            if self.frame:
                cv.imshow(self.window_title, self.frame)

                # print(f"Port:{widget.port} - fps:{widget.fps}")

            if cv.waitKey(1) == ord("q"):
                self.stop()

    def stop(self):
        self.stopped = True
        cv.destroyAllWindows()
    