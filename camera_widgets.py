# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:04:30 2024

@author: Wittke
"""
import os
from utilities import (
    Box,
    calculate_frame_box_static,
    calculate_ranking,
    get_processed_frame,
    os_sensitive_backslashes,
)
from detection_widgets import HumanWidget, DeepFaceWidget, FaceWidget, DetectionWidget
from multiprocessing import Process
import cv2 as cv
from logger import Logger
from timeit import default_timer as timer
from output_widgets import ImageShowWidget
import numpy as np
from shut_down_coordinator import Shutdown_Coordinator


class CameraWidget:
    frame: np.ndarray

    def __init__(
        self,
        port: int,
        resolution,
        camera_fps: int,
        l=Logger(),
        sc=Shutdown_Coordinator(),
    ):

        self.sc = sc
        self.port = port
        self.resolution = resolution
        self.camera_fps = camera_fps
        self.l = l
        self.frame = None
        self.frame_counter = 0
        self.fps = 0
        self.l.passing(f"Creating CameraWidget {self.port}")
        if os.name == "nt":
            self.cap = cv.VideoCapture(port, cv.CAP_DSHOW)
        else:
            self.cap = cv.VideoCapture(port)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.resolution[1])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.resolution[0])
        self.cap.set(cv.CAP_PROP_FPS, self.camera_fps)

        self.grabbed = False
        self.stopped = True
        self.process = None

    def start(self):
        if self.stopped:
            self.l.passing(f"Starting CameraWidget {self.port}")
            self.stopped = False
            self.process = Process(target=self.run)
            self.process.start()

    def run(self):
        time_start = timer()
        frame_counter = 0
        try:
            while not self.stopped:
                if not self.sc.running():
                    self.l.warning("Shutdown Detected exiting")
                    self.stop()
                    break
                self.grabbed, self.frame = self.cap.read()
                if self.grabbed:
                    frame_counter += 1
                    time_end = timer()

                    if time_end - time_start > 1:
                        self.fps = frame_counter
                        time_start = timer()
                        frame_counter = 0
        except Exception as e:
            self.l.error(e.with_traceback(e.__traceback__))
            self.stop()
            raise e

    def stop(self):
        self.stopped = True
        self.l.warning(f"Stopping CameraWidget {self.port}")
        self.sc.stop()
        self.cap.release()
        if self.process is not None:
            self.process.terminate()

if __name__ == "__main__":

    l = Logger(True)
    l.passingblue("Starting Minimum example, only used for debugging purposes")
    human_detection_path = os_sensitive_backslashes("models/detection/yolov8n.pt")
    face_detection_path = os_sensitive_backslashes("models/face/yolov8n-face.pt")
    database_path = os_sensitive_backslashes("database")

    captures = []
    ports = [0]
    min_ex_show = []

    for i, port in enumerate(ports):
        l.passing("Creating VidCaps")
        captures.append(
            CameraWidget(
                port,
                [720, 1280],
                30,
                l,
            )
        )
        min_ex_show.append(ImageShowWidget(f"Minimum example Cap {port}", l))
        min_ex_show[i].start()
        captures[i].start()

    box: Box = None
    try:
        while True:
            for i, port in enumerate(ports):

                cap: CameraWidget = captures[i]

                if cap.grabbed:
                    boxes = cap.get_detection_bounds()
                    if boxes == []:
                        l.warning("Boxes empty, continuing")
                    else:
                        box = calculate_frame_box_static(boxes)
                    if box is None:
                        cropped_frame = cap.frame
                    else:
                        cropped_frame = get_processed_frame(box, cap.frame)
                    min_ex_show[i].update_frame(cropped_frame)
                    min_ex_show[i].show_image()
                else:
                    l.warning("No frame returned")

    except (KeyboardInterrupt, Exception) as e:
        l.error(f"{e}\nStopping widgets")
        for widget in min_ex_show:
            widget.stop()
        for cap in captures:
            cap.stop()
        exit()
