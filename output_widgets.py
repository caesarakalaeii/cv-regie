from abc import ABC
import os
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
        self.l.warning('OpenCV does not support threading. \nNo thread will be started, use show_image and update_frame instead')

    
    def run(self):
        self.l.passingblue("Starting output widget")

        while not self.stopped:
            self.show_image()
       
    def show_image(self):     
        if self.frame is None:
                #self.l.info("no frame recieved")
                return
        try:
            cv.imshow(self.window_title, self.frame)
        except Exception as e:
            self.l.error(e) 
        # print(f"Port:{widget.port} - fps:{widget.fps}")

        if cv.waitKey(1) == ord("q"):
            self.stop()
    
    def stop(self):
        self.l.warning('Stopping Output Widget')
        self.stopped = True
        cv.destroyAllWindows()
        
       
  
if __name__ == '__main__':
    
    l = Logger(True)
    l.passingblue("Starting Minimum example, only used for debugging purposes")
    
    captures = []
    
    ports = [0, 8]
    min_ex = []
    
    for i, port in enumerate(ports):
        l.passing("Creating VidCaps")
        if os.name == 'nt':
            captures.append(cv.VideoCapture(port, cv.CAP_DSHOW))
        else:
            captures.append(cv.VideoCapture(port))
        min_ex.append(ImageShowWidget(f'Minimum example Cap {port}', l))
        min_ex[i].start()
    
    while True:
        for i, port in enumerate(ports):
            cap: cv.VideoCapture = captures[i]
            grabbed, frame = cap.read()
            
            if grabbed:
                min_ex[i].update_frame(frame)
                min_ex[i].show_image()
            else:
                l.warning("No frame returned")