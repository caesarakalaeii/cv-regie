# -*- coding: utf-8 -*-
"""
Created on Thu May  2 08:46:22 2024

@author: Wittke
"""

from threading import Thread
import cv2 as cv
import time

class Camera_Widget:
    
    def __init__(self, port=0, res=[720,1280], fps=30):
        self.port = port
        self.fps = fps
        self.cap = cv.VideoCapture(port)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, res[1])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, res[0])
        self.cap.set(cv.CAP_PROP_FPS, fps)
        
    def set_all_widgets(self,all_widgets):
        self.all_widgets = all_widgets
        
    def start(self):
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        Thread(target=self.run, daemon=True).start()
        Thread(target=self.show_frame, daemon=True).start()
        
    def run(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.cap.read()
                time.sleep(1 / self.fps)
                
    def stop(self):
        self.stopped = True
        self.cap.release()
        cv.destroyAllWindows()
    
    def show_frame(self):
        
        while not self.stopped:
            if self.grabbed:
                cv.imshow("camera_{0}".format(self.port), self.frame)
                if cv.waitKey(1) == ord("q"):
                    for widget in self.all_widgets:
                        widget.stop()
                        
if __name__ == '__main__':
    
    ports = [8]
    widgets = []
    
    for port in ports:
        widgets.append(Camera_Widget(port))
       
    for widget in widgets:
        widget.set_all_widgets(widgets)
        widget.start()
        
    
