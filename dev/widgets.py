# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:04:30 2024

@author: Wittke
"""

from threading import Thread
import cv2 as cv
from timeit import default_timer as timer
from ultralytics import YOLO

class ImageShowWidget():
    "Main widget to show all camera feeds"

    def __init__(self,
                 ports: [..., int],
                 resolution: [int, int],
                 camera_fps: int,
                 human_detection_path: str,
                 face_detection_path: str):

        self.ports = ports
        self.resolution = resolution
        self.camera_fps = camera_fps

        self.human_detection_path = human_detection_path
        self.face_detection_path = face_detection_path

        self.cameraWidgets = []

        self.stopped = False

        self.start()

    def start(self):

        for port in self.ports:
            widget = CameraWidget(port, self.resolution, self.camera_fps, self.human_detection_path, self.face_detection_path)
            self.cameraWidgets.append(widget)

        for widget in self.cameraWidgets:
            widget.start()

        self.run()

    def run(self):

        while not self.stopped:
            for widget in self.cameraWidgets:
                if widget.grabbed:
                    frame = widget.frame
                    cv.imshow(f"Camera-{widget.port}-normal", frame)

                if widget.human_detection:
                    frame = widget.human_detection_frame
                    cv.imshow(f"Camera-{widget.port}-human", frame)

                if widget.face_detection:
                    frame = widget.face_detection_frame
                    cv.imshow(f"Camera-{widget.port}-face", frame)

                    #print(f"Port:{widget.port} - fps:{widget.fps}")

                if cv.waitKey(1) == ord("q"):
                    self.stop()

    def stop(self):
        self.stopped = True
        for widget in self.cameraWidgets:
            widget.stop()
        cv.destroyAllWindows()


class CameraWidget():

    def __init__(self,
                 port: int,
                 resolution: [int, int],
                 camera_fps: int,
                 human_detection_path: str,
                 face_detection_path: str):

        self.port = port
        self.resolution = resolution
        self.camera_fps = camera_fps

        self.human_detection_path = human_detection_path
        self.human_detection_started = False
        self.human_detection = False
        self.human_detection_widget = None
        self.human_detection_frame = None

        self.face_detection_path = face_detection_path
        self.face_detection_started = False
        self.face_detection = False
        self.face_detection_widget = None
        self.face_detection_frame = None

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

    def start(self):
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed:
            self.human_detection_widget = HumanWidget(self, self.human_detection_path)
            self.face_detection_widget = FaceWidget(self, self.face_detection_path)
            self.thread.start()

    def run(self):
        time_start = timer()
        frame_counter = 0
        while not self.stopped:
            if self.grabbed:
                if not self.human_detection_started:
                    self.human_detection_widget.start()
                    self.human_detection_started = True

                if not self.face_detection_started:
                    self.face_detection_widget.start()
                    self.face_detection_started = True

                self.grabbed, self.frame = self.cap.read()
                frame_counter += 1
                time_end = timer()

                if time_end - time_start > 1:
                    self.fps = frame_counter
                    time_start = timer()
                    frame_counter = 0
            else:
                self.stop()

    def stop(self):
        self.stopped = True
        self.human_detection_widget.stop()
        self.face_detection_widget.stop()
        self.cap.release()


class HumanWidget():

    def __init__(self,
                 widget: CameraWidget,
                 human_detection_path: str):

        self.widget = widget

        self.human_detection = False
        self.human_detection_path = human_detection_path

        self.model = YOLO(human_detection_path)

        self.widget_frame = None
        self.human_detection_frame = None

        self.thread = Thread(target=self.run)
        self.stopped = False

    def start(self):
        self.thread.start()

    def run(self):
        while not self.stopped:
            if self.widget.grabbed:
                self.widget_frame = self.widget.frame
                result = self.model.track(self.widget_frame,
                                          tracker="bytetrack.yaml",
                                          imgsz=320,
                                          classes=[0],
                                          verbose=False)
                self.human_detection_frame = result[0].plot()
                self.widget.human_detection_frame = self.human_detection_frame

                self.human_detection = True
                self.widget.human_detection = self.human_detection

    def stop(self):
        self.stopped = True


class FaceWidget():

    def __init__(self,
                 widget: CameraWidget,
                 face_detection_path: str):

        self.widget = widget

        self.face_detection = False
        self.face_detection_path = face_detection_path

        self.model = YOLO(face_detection_path)

        self.widget_frame = None
        self.face_detection_frame = None

        self.thread = Thread(target=self.run)
        self.stopped = False

    def start(self):
        self.thread.start()

    def run(self):
        while not self.stopped:
            if self.widget.grabbed:
                self.widget_frame = self.widget.frame
                result = self.model.track(self.widget_frame,
                                          tracker="bytetrack.yaml",
                                          imgsz=320,
                                          classes=[0],
                                          verbose=False)
                self.face_detection_frame = result[0].plot()
                self.widget.face_detection_frame = self.face_detection_frame

                self.face_detection = True
                self.widget.face_detection = self.face_detection

    def stop(self):
        self.stopped = True


if __name__ == "__main__":
    ports = [0]
    resolution = [720, 1280]
    camera_fps = 30

    human_detection_path = "models/detection/yolov8n.pt"
    face_detection_path = "models/face/yolov8n-face.pt"

    imageShow = ImageShowWidget(ports,
                                resolution,
                                camera_fps,
                                human_detection_path,
                                face_detection_path)