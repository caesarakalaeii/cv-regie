# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:04:30 2024

@author: Wittke
"""

from threading import Thread
import cv2 as cv
from timeit import default_timer as timer
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np


class ImageShowWidget():
    "Main widget to show all camera feeds"

    def __init__(self,
                 ports: [..., int],
                 resolution: [int, int],
                 camera_fps: int,
                 human_detection_path: str,
                 face_detection_path: str,
                 database_path: str):

        self.ports = ports
        self.resolution = resolution
        self.camera_fps = camera_fps

        self.human_detection_path = human_detection_path
        self.face_detection_path = face_detection_path
        self.database_path = database_path

        self.cameraWidgets = []

        self.stopped = False

        self.start()

    def start(self):

        for port in self.ports:
            widget = CameraWidget(port,
                                  self.resolution,
                                  self.camera_fps,
                                  self.human_detection_path,
                                  self.face_detection_path,
                                  self.database_path)
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
                 face_detection_path: str,
                 database_path):

        self.port = port
        self.resolution = resolution
        self.camera_fps = camera_fps

        self.human_detection_path = human_detection_path
        self.human_detection_started = False
        self.human_detection = False
        self.human_detection_widget = None
        self.human_detection_frame = None
        self.human_detection_score = 0
        self.human_detection_data = [[],[],[],[],[]]

        self.face_detection_path = face_detection_path
        self.face_detection_started = False
        self.face_detection = False
        self.face_detection_widget = None
        self.face_detection_frame = None
        self.face_detection_score = 0
        self.face_detection_score = 0
        self.face_detection_data = [[],[],[],[],[]]

        self.database_path = database_path
        self.deepface_detection_started = False
        self.deepface_detection = False
        self.deepface_detection_widget = None
        self.deepface_detection_frame = None
        self.deepface_detections_data = [[],[],[],[],[]]

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
            self.deepface_detection_widget = DeepFaceWidget(self, self.database_path)
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

                if not self.deepface_detection_started:
                    self.deepface_detection_widget.start()
                    self.deepface_detection_started = True

                self.grabbed, self.frame = self.cap.read()
                frame_counter += 1
                time_end = timer()

                if time_end - time_start > 1:
                    self.fps = frame_counter
                    time_start = timer()
                    frame_counter = 0

                print(self.human_detection_data)

            else:
                self.stop()

    def stop(self):
        self.stopped = True
        self.human_detection_widget.stop()
        self.face_detection_widget.stop()
        self.deepface_detection_widget.stop()
        self.cap.release()


class HumanWidget():

    def __init__(self,
                 widget: CameraWidget,
                 human_detection_path: str):

        self.widget = widget

        self.human_detection = False
        self.human_detection_path = human_detection_path
        self.human_detection_score = 0
        self.human_detection_data = [[],[],[],[],[]]

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

                self.human_detection_score = self.countIDs(result)

                self.human_detection_data = self.getResultData(result)

                self.human_detection_frame = result[0].plot()
                self.widget.human_detection_frame = self.human_detection_frame

                self.human_detection = True
                self.widget.human_detection = self.human_detection
                self.widget.human_detection_score = self.human_detection_score
                self.widget.human_detection_data = self.human_detection_data

    def stop(self):
        self.stopped = True

    def countIDs(self,
                 result):
        counts = 0
        if result[0].boxes.id is not None:
            counts = max(result[0].boxes.id.int().cpu().tolist())
        return counts

    def getResultData(self,
                      result):
        if result[0].boxes.id is not None:
            for i,identity in enumerate(result[0].boxes.id):
                x = result[0].boxes.xywh[i][0]
                y = result[0].boxes.xywh[i][1]
                w = result[0].boxes.xywh[i][2]
                h = result[0].boxes.xywh[i][3]

                if i == 0:
                    data = [[int(identity)],[int(x)],[int(y)],[int(h)],[int(w)]]
                else:
                    data.append([[int(identity)],[int(x)],[int(y)],[int(h)],[int(w)]])

class FaceWidget():

    def __init__(self,
                 widget: CameraWidget,
                 face_detection_path: str):

        self.widget = widget

        self.face_detection = False
        self.face_detection_path = face_detection_path
        self.face_detection_score = 0
        self.face_detection_data = [[],[],[],[],[]]

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

                self.face_detection_score = self.countIDs(result)

                self.face_detection_data = self.getResultData(result)

                self.face_detection_frame = result[0].plot()
                self.widget.face_detection_frame = self.face_detection_frame

                self.face_detection = True
                self.widget.face_detection = self.face_detection
                self.widget.face_detection_score = self.face_detection_score
                self.widget.face_detection_data = self.face_detection_data

    def stop(self):
        self.stopped = True

    def countIDs(self,
                 result):
        counts = 0
        if result[0].boxes.id is not None:
            counts = max(result[0].boxes.id.int().cpu().tolist())
        return counts

    def getResultData(self,
                      result):
        if result[0].boxes.id is not None:
            for i,identity in enumerate(result[0].boxes.id):
                x = result[0].boxes.xywh[i][0]
                y = result[0].boxes.xywh[i][1]
                w = result[0].boxes.xywh[i][2]
                h = result[0].boxes.xywh[i][3]
            
                if i == 0:
                    data = [[identity],[x],[y],[h],[w]]
                else:
                    data.append([[identity],[x],[y],[h],[w]])

class DeepFaceWidget():

    def __init__(self,
                 widget: CameraWidget,
                 database_path: str):

        self.widget = widget

        self.database_path = database_path

        self.deepface_detections_data = [[],[],[],[],[]]

        self.widget_frame = None

        self.thread = Thread(target=self.run)
        self.stopped = False

    def start(self):
        self.thread.start()

    def run(self):
        while not self.stopped:
            if self.widget.grabbed:
                self.widget_frame = self.widget.frame
                dataframes = DeepFace.find(img_path = np.array(self.widget_frame),
                                           db_path = self.database_path,
                                           enforce_detection=False,
                                           silent=True,
                                           detector_backend = "yolov8",
                                           distance_metric = "euclidean_l2"
                                           )

                self.deepface_detections_data = self.getResultData(dataframes)

                self.widget.deepface_detections_data = self.deepface_detections_data

    def stop(self):
        self.stopped = True

    def getResultData(self,
                      dataframes):
        if len(dataframes) > 0:
            for i,entry in enumerate(dataframes):
                if not entry.empty:
                    identity = entry["identity"][0][len(database_path)+1:-6]
                    x = entry["source_x"][0]
                    y = entry["source_y"][0]
                    w = entry["source_w"][0]
                    h = entry["source_h"][0]

                    if i == 0:
                        data = [[identity], [x], [y], [h], [w]]
                    else:
                        data.append([[identity], [x], [y], [h], [w]])
                        
if __name__ == "__main__":
    ports = [1]
    resolution = [720, 1280]
    camera_fps = 30

    human_detection_path = "models/detection/yolov8n.pt"
    face_detection_path = "models/face/yolov8n-face.pt"
    database_path = "./database"

    imageShow = ImageShowWidget(ports,
                                resolution,
                                camera_fps,
                                human_detection_path,
                                face_detection_path,
                                database_path)
