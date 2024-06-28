# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:04:30 2024

@author: Wittke

identity_data is a data struct per cam-widget, that includs the id given by deepface and a boudning box
if the id is -1, that means the yolo can see more persons as deepface can identify
the id is the number in the deepface database, so it needs to bee a int and NOT a name


"""

from threading import Thread
import cv2 as cv
from timeit import default_timer as timer
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np


class ImageShowWidget:
    "Main widget to show all camera feeds"

    def __init__(
        self,
        ports: [..., int],
        resolution: [int, int],
        camera_fps: int,
        pose_detection_path: str,
        database_path: str,
    ):

        self.ports = ports
        self.resolution = resolution
        self.camera_fps = camera_fps

        self.pose_detection_path = pose_detection_path
        self.database_path = database_path

        self.cameraWidgets = []

    def start(self):

        for port in self.ports:
            widget = CameraWidget(
                port,
                self.resolution,
                self.camera_fps,
                self.pose_detection_path,
                self.database_path,
            )
            self.cameraWidgets.append(widget)

        for widget in self.cameraWidgets:
            widget.start()

        self.run()

    def run(self):
        self.stopped = False

        while not self.stopped:
            for widget in self.cameraWidgets:
                if widget.grabbed:
                    frame = widget.frame
                    cv.imshow(f"Camera-{widget.port}-normal", frame)

                if widget.pose_detection:
                    frame = widget.pose_detection_frame
                    cv.imshow(f"Camera-{widget.port}-pose", frame)

                    # print(f"Port:{widget.port} - fps:{widget.fps}")

                if cv.waitKey(1) == ord("q"):
                    self.stop()

    def stop(self):
        self.stopped = True
        for widget in self.cameraWidgets:
            widget.stop()
        cv.destroyAllWindows()


class CameraWidget:

    def __init__(
        self,
        port: int,
        resolution: [int, int],
        camera_fps: int,
        pose_detection_path: str,
        database_path,
    ):

        self.port = port
        self.resolution = resolution
        self.camera_fps = camera_fps

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

        self.identity_data = np.array([[0, 0, 0, 0, 0]])

    def start(self):
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed:
            self.pose_detection_widget = PoseWidget(self, self.pose_detection_path)
            self.deepface_detection_widget = DeepFaceWidget(self, self.database_path)
            self.thread.start()

    def run(self):
        time_start = timer()
        frame_counter = 0
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
        self.pose_detection_widget.stop()
        self.deepface_detection_widget.stop()
        self.cap.release()


class PoseWidget:

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
            if self.widget.grabbed:
                self.widget_frame = self.widget.frame
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
                self.widget.pose_detection_frame = self.pose_detection_frame

                self.pose_detection = True
                self.widget.pose_detection = self.pose_detection
                self.widget.pose_detection_score = self.pose_detection_score
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


class DeepFaceWidget:

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
            if self.widget.grabbed:
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
                    identity = int(entry["identity"][0][len(database_path) + 1 : -6])
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


if __name__ == "__main__":
    ports = [0]
    resolution = [720, 1280]
    camera_fps = 30

    pose_detection_path = "models/detection/yolov8n-pose.pt"
    database_path = "./database"

    imageShow = ImageShowWidget(
        ports, resolution, camera_fps, pose_detection_path, database_path
    )
    imageShow.start()
