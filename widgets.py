# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:04:30 2024

@author: Wittke
"""
from utilities import identity_from_string, calculate_ranking, pad_to_16by9, Box
from detection_widgets import HumanWidget, DeepFaceWidget, FaceWidget, DetectionWidget
from threading import Thread
import cv2 as cv
from timeit import default_timer as timer






class CameraWidget:
    
    widgets: [DetectionWidget]

    def __init__(
        self,
        port: int,
        resolution: (int, int),
        camera_fps: int,
        human_detection_path: str,
        face_detection_path: str,
        database_path,
    ):

        self.port = port
        self.resolution = resolution
        self.camera_fps = camera_fps

        self.frame = None
        self.frame_counter = 0
        self.fps = 0

        self.cap = cv.VideoCapture(self.port)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.resolution[1])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.resolution[0])
        self.cap.set(cv.CAP_PROP_FPS, self.camera_fps)

        self.grabbed = False

        self.thread = Thread(target=self.run)
        self.stopped = False
        human_detection_widget = HumanWidget(self, human_detection_path)
        face_detection_widget = FaceWidget(self, face_detection_path)
        deepface_detection_widget = DeepFaceWidget(self, database_path)
        self.widgets = [
            human_detection_widget,
            face_detection_widget,
            deepface_detection_widget
            ]
        

    def start(self):
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
        for widget in self.widgets:
            if not widget.stopped:
                widget.start()
    
    def update_widgets(self):
        for widget in self.widgets:
            if not widget.stopped:
                widget.update(self.frame)
    
    def stop(self):
        self.stopped = True
        for widget in self.widgets:
            if not widget.stopped:
                widget.stop()
        self.cap.release()

    def get_ranking(self):
        return calculate_ranking(
            self.frame.shape, self.widgets
        )
        
    def get_detection_bounds(self):
        return self.human_detection_widget.getResultData()






if __name__ == "__main__":
    ports = [1]
    resolution = [720, 1280]
    camera_fps = 30

    human_detection_path = "models/detection/yolov8n.pt"
    face_detection_path = "models/face/yolov8n-face.pt"
    database_path = "./database"

    imageShow = ImageShowWidget(
        ports,
        resolution,
        camera_fps,
        human_detection_path,
        face_detection_path,
        database_path,
    )
