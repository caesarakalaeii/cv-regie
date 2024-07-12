# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:04:30 2024

@author: Wittke
"""
from utilities import calculate_ranking
from detection_widgets import HumanWidget, DeepFaceWidget, FaceWidget, DetectionWidget
from threading import Thread
import cv2 as cv
from logger import Logger
from timeit import default_timer as timer






class CameraWidget:
    
    widgets: list[DetectionWidget]

    def __init__(
        self,
        port: int,
        resolution: list[int, int],
        camera_fps: int,
        human_detection_path: str,
        face_detection_path: str,
        database_path,
        l= Logger()
    ):

        
        self.port = port
        self.resolution = resolution
        self.camera_fps = camera_fps
        self.l = l
        self.frame = None
        self.frame_counter = 0
        self.fps = 0
        self.l.passing(f'Creating CameraWidget {self.port}')
        self.cap = cv.VideoCapture(self.port)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.resolution[1])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.resolution[0])
        self.cap.set(cv.CAP_PROP_FPS, self.camera_fps)

        
        human_detection_widget = HumanWidget(human_detection_path)
        face_detection_widget = FaceWidget(face_detection_path)
        deepface_detection_widget = DeepFaceWidget(database_path)
        self.widgets = [
            human_detection_widget,
            face_detection_widget,
            deepface_detection_widget
            ]
        
        
        self.grabbed = False
        self.thread = Thread(target=self.run)
        self.stopped = False
        

    def start(self):
        self.l.passing(f'Starting CameraWidget {self.port}')
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed:
            self.thread.start()

    def run(self):
        time_start = timer()
        frame_counter = 0
        while not self.stopped:
            if self.grabbed:
                
                self.init_widgets()
                
                self.grabbed, self.frame = self.cap.read()
                self.update_widgets()
                frame_counter += 1
                time_end = timer()

                if time_end - time_start > 1:
                    self.fps = frame_counter
                    time_start = timer()
                    frame_counter = 0

            else:
                self.stop()

    def init_widgets(self):
        widget: DetectionWidget
        for widget in self.widgets:
            if widget.stopped:
                self.l.passing(f'Starting DetectionWidget {widget.widget_type} for Camera {self.port}')
                widget.start()
    
    def update_widgets(self):
        widget: DetectionWidget
        for widget in self.widgets:
            if not widget.stopped:
                widget.update_frame(self.frame)
    
    def stop(self):
        widget: DetectionWidget
        self.stopped = True
        for widget in self.widgets:
            if not widget.stopped:
                widget.stop()
        self.cap.release()

    def get_ranking(self):
        human_detection_widget:DetectionWidget = self.widgets[0]
        persons = human_detection_widget.countIDs()
        face_detection_widget:DetectionWidget = self.widgets[1]
        faces = face_detection_widget.countIDs()
        
        return calculate_ranking(
            self.frame.shape, persons, faces
        )
        
    def get_detection_bounds(self):
        return self.widgets[0].getResultData()






