# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:38:18 2024

@author: Wittke
"""

import cv2 as cv

from thread.cameraThread import CameraWidget
from thread.directionThread import DirectionWidget

class LeadWidget:
    "Main widget to show all camera feeds"

    def __init__(
        self,
        ports: [..., int],
        resolution: [int, int],
        camera_fps: int,
        skipped_frames: [int, int],
        director_fartique: int,
        pose_detection_path: str,
        database_path: str,
        verbose: bool = True,
        picturesque: int = 0
    ):

        self.ports = ports
        self.resolution = resolution
        self.camera_fps = camera_fps
        self.skipped_frames = skipped_frames
        self.director_fartique = director_fartique

        self.pose_detection_path = pose_detection_path
        self.database_path = database_path

        self.verbose = verbose
        self.picturesque = picturesque

        self.cameraWidgets = []
        self.frameObjects = []

    def start(self):

        for port in self.ports:
            widget = CameraWidget(
                port,
                self.resolution,
                self.camera_fps,
                self.skipped_frames,
                self.pose_detection_path,
                self.database_path,
            )
            self.cameraWidgets.append(widget)

        for widget in self.cameraWidgets:
            widget.start()
            self.frameObjects.append(widget.frameObject)

        self.directionWidget = DirectionWidget(self.frameObjects,
                                               self.director_fartique)
        self.directionWidget.start()

        self.run()

    def run(self):
        self.stopped = False

        while not self.stopped:
            for widget in self.cameraWidgets:
                if widget.grabbed and self.picturesque >= 1 and widget.frameObject.frame is not None:
                    cv.imshow(f"Camera-{widget.port}-normal", widget.frameObject.frame)

                if widget.pose_detection and self.picturesque >= 2 and  widget.frameObject.yoloFrame is not None:
                    cv.imshow(f"Camera-{widget.port}-pose", widget.frameObject.yoloFrame)

                if self.verbose:
                    print(f"Port:{widget.port} - fps:{widget.frameObject.fps}")

            if self.directionWidget.bestIndex is not None:
               cv.imshow(f"Best frame", self.cameraWidgets[self.directionWidget.bestIndex].frameObject.frame)

            if cv.waitKey(1) == ord("q"):
                self.stop()

    def stop(self):
        self.stopped = True
        self.directionWidget.stop()
        for widget in self.cameraWidgets:
            widget.stop()
        cv.destroyAllWindows()
