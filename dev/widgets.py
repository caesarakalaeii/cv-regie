# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:04:30 2024

@author: Wittke
"""

from threading import Thread
import cv2 as cv
import numpy as np
from timeit import default_timer as timer


class ImageShowWidget():
    "Main widget to show all camera feeds"

    def __init__(self,
                 ports: [..., int],
                 resolution: [int, int],
                 camera_fps: int):

        self.ports = ports
        self.resolution = resolution
        self.camera_fps = camera_fps

        self.cameraWidgets = []

        self.stopped = False

        self.start()

    def start(self):

        for port in self.ports:
            widget = CameraWidget(port, self.resolution, self.camera_fps)
            self.cameraWidgets.append(widget)

        for widget in self.cameraWidgets:
            widget.start()

        self.run()

    def run(self):

        while not self.stopped:
            for widget in self.cameraWidgets:
                if widget.grabbed:
                    frame = widget.frame
                    cv.imshow(f"Camera-{widget.port}", frame)

                    print(f"Port:{widget.port} - fps:{widget.fps}")

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
                 camera_fps: int):

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

    def start(self):
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed:
            self.thread.start()

    def run(self):
        time_start = timer()
        frame_counter = 0
        while not self.stopped:
            if self.grabbed:
                self.grabbed, self.frame = self.cap.read()
                frame_counter += 1
                time_end =timer()

                if time_end - time_start > 1:
                    self.fps = frame_counter
                    time_start = timer()
                    frame_counter = 0
            else:
                self.stop()

    def stop(self):
        self.stopped = True
        self.cap.release()

class HumanWidget():

    def __init__(self):
        pass

    def start(self):
        pass

    def run(self):
        pass

    def stop(self):
        pass


class FaceWidget():

    def __init__(self):
        pass

    def start(self):
        pass

    def run(self):
        pass

    def stop(self):
        pass

if __name__ == "__main__":
    ports = [0]
    resolution = [720, 1280]
    camera_fps = 30

    imageShow = ImageShowWidget(ports, resolution, camera_fps)