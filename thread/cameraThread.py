# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:39:35 2024

@author: Wittke
"""

import numpy as np
import cv2 as cv

from threading import Thread
from ultralytics import YOLO
from deepface import DeepFace
from timeit import default_timer as timer


class Frame:

    def __init__(self):

        self.frame = None
        self.yoloFrame = None
        self.fps = 0
        self.pose_detection_score = 0
        self.face_detection_score = 0
        self.identification_detection_score = 0


class CameraWidget:

    def __init__(
        self,
        port: int,
        resolution: [int, int],
        camera_fps: int,
        skipped_frames: [int, int],
        pose_detection_path: str,
        database_path,
    ):

        self.port = port
        self.resolution = resolution
        self.camera_fps = camera_fps
        self.skipped_frames = skipped_frames

        self.frameObject = Frame()

        self.pose_detection_path = pose_detection_path
        self.pose_detection_started = False
        self.pose_detection = False
        self.pose_detection_widget = None
        self.pose_detection_frame = None
        self.pose_detection_score = 0
        self.pose_detection_data = np.array([[0, 0, 0, 0, 0]])
        self.pose_detection_keypoints = np.array([[[0, 0], [0, 0], [0, 0]]])

        self.database_path = database_path
        self.deepface_detection_started = False
        self.deepface_detection = False
        self.deepface_detection_widget = None
        self.deepface_detection_frame = None
        self.deepface_detections_data = np.array([[0, 0, 0]])

        self.frame_counter = 0

        self.cap = cv.VideoCapture(self.port)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.resolution[1])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.resolution[0])
        self.cap.set(cv.CAP_PROP_FPS, self.camera_fps)

        self.grabbed = False

        self.thread = Thread(target=self.run)
        self.stopped = False

        self.identity_data = np.array([[0, 0, 0, 0, 0]])

    def start(self):
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed:
            self.pose_detection_widget = YoloWidget(self, self.pose_detection_path)
            self.deepface_detection_widget = DeepfaceWidget(self, self.database_path)
            self.thread.start()

    def run(self):
        time_start = timer()
        counter = 0
        while not self.stopped:
            if self.grabbed:
                if not self.pose_detection_started:
                    self.pose_detection_widget.start()
                    self.pose_detection_started = True

                if not self.deepface_detection_started:
                    self.deepface_detection_widget.start()
                    self.deepface_detection_started = True

                self.identity_data = np.copy(self.pose_detection_data)
                self.identity_data[:, 0] = -1

                for identity in self.deepface_detections_data:
                    center = identity[1:]
                    euclid_norm = np.sum(
                        np.sqrt(
                            np.sum(
                                np.power(center - self.pose_detection_keypoints, 2),
                                axis=2,
                            )
                        ),
                        axis=1,
                    )
                    min_idx = np.argmin(euclid_norm)
                    self.identity_data[min_idx][0] = identity[0]

                self.frameObject.pose_detection_score = (self.pose_detection_keypoints.size)/2
                self.frameObject.face_detection_score = (self.pose_detection_keypoints.size - np.count_nonzero(self.pose_detection_keypoints))/2
                self.frameObject.identification_detection_score = len(np.where(self.identity_data[:, 0] == -1))

                self.grabbed, self.frameObject.frame = self.cap.read()
                counter += 1
                self.frame_counter += 1
                time_end = timer()

                if time_end - time_start > 1:
                    self.frameObject.fps = counter
                    time_start = timer()
                    counter = 0

            else:
                self.stop()

    def stop(self):
        self.stopped = True
        self.pose_detection_widget.stop()
        self.deepface_detection_widget.stop()
        self.cap.release()


class YoloWidget:

    def __init__(self, widget: CameraWidget, pose_detection_path: str):

        self.widget = widget

        self.pose_detection = False
        self.pose_detection_path = pose_detection_path
        self.pose_detection_score = 0
        self.pose_detection_data = np.array([[0, 0, 0, 0, 0]])
        self.pose_detection_keypoints = np.array([[[0, 0], [0, 0], [0, 0]]])

        self.model = YOLO(pose_detection_path)

        self.widget_frame = None
        self.pose_detection_frame = None

        self.thread = Thread(target=self.run)
        self.stopped = False

    def start(self):
        self.thread.start()

    def run(self):
        while not self.stopped:
            if self.widget.grabbed and self.widget.frame_counter % self.widget.skipped_frames[0] == 0:
                self.widget_frame = self.widget.frameObject.frame
                result = self.model.track(
                    self.widget_frame,
                    tracker="bytetrack.yaml",
                    imgsz=320,
                    classes=[0],
                    verbose=False,
                )

                self.pose_detection_score = self.countIDs(result)

                self.pose_detection_data, self.pose_detection_keypoints = (
                    self.getResultData(result)
                )

                self.pose_detection_frame = result[0].plot()
                self.widget.frameObject.yoloFrame = self.pose_detection_frame

                self.pose_detection = True
                self.widget.pose_detection = self.pose_detection
                self.widget.frameObject.pose_detection_score = self.pose_detection_score
                self.widget.pose_detection_data = self.pose_detection_data
                self.widget.pose_detection_keypoints = self.pose_detection_keypoints

    def stop(self):
        self.stopped = True

    def countIDs(self, result):
        counts = 0
        if result[0].boxes.id is not None:
            counts = max(result[0].boxes.id.int().cpu().tolist())
        return counts

    def getResultData(self, result):
        data = np.array([[0, 0, 0, 0, 0]])
        keypoints = np.array([[[0, 0], [0, 0], [0, 0]]])

        if result[0].boxes.id is not None:
            for i, identity in enumerate(result[0].boxes.id):
                identity = int(identity)
                x = int(result[0].boxes.xywh[i][0])
                y = int(result[0].boxes.xywh[i][1])
                w = int(result[0].boxes.xywh[i][2])
                h = int(result[0].boxes.xywh[i][3])

                points = result[0].keypoints.xy[0]
                nose = [int(points[0][0]), int(points[0][1])]
                left_eye = [int(points[1][0]), int(points[1][1])]
                right_eye = [int(points[2][0]), int(points[2][1])]

                if i == 0:
                    data = np.array([[identity, x, y, h, w]])
                    keypoints = np.array([[nose, left_eye, right_eye]])
                else:
                    data = np.append(data, np.array([[identity, x, y, h, w]]), axis=0)
                    keypoints = np.append(
                        keypoints, np.array([[nose, left_eye, right_eye]]), axis=0
                    )
        return data, keypoints


class DeepfaceWidget:

    def __init__(self, widget: CameraWidget, database_path: str):

        self.widget = widget

        self.database_path = database_path

        self.deepface_detections_data = np.array([[0, 0, 0]])

        self.widget_frame = None

        self.thread = Thread(target=self.run)
        self.stopped = False

    def start(self):
        self.thread.start()

    def run(self):
        while not self.stopped:
            if self.widget.grabbed and self.widget.frame_counter % self.widget.skipped_frames[1] == 0:
                self.widget_frame = self.widget.frame
                dataframes = DeepFace.find(
                    img_path=np.array(self.widget_frame),
                    db_path=self.database_path,
                    enforce_detection=False,
                    silent=True,
                    detector_backend="yolov8",
                    distance_metric="euclidean_l2",
                )

                self.deepface_detections_data = self.getResultData(dataframes)

                self.widget.deepface_detections_data = self.deepface_detections_data

    def stop(self):
        self.stopped = True

    def getResultData(self, dataframes):
        data = np.array([[0, 0, 0]])
        if len(dataframes) > 0:
            for i, entry in enumerate(dataframes):
                if not entry.empty:
                    identity = int(entry["identity"][0][len(self.database_path) + 1 : -6])
                    x = entry["source_x"][0]
                    y = entry["source_y"][0]
                    w = entry["source_w"][0]
                    h = entry["source_h"][0]

                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)

                    if i == 0:
                        data = np.array([[identity, center_x, center_y]])

                    else:
                        data = np.append(data, [[identity, center_x, center_y]], axis=0)

        return data