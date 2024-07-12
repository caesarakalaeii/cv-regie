from abc import ABC

from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
from threading import Thread
from utilities import Box, identity_from_string

class DetectionWidget (ABC):
    
    widget_frame: np.ndarray
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def countIDs(self):
        pass

    def getResultData(self):
        pass
    
    def update_frame(self, frame):
        self.widget_frame = frame



class HumanWidget(DetectionWidget):

    def __init__(self, human_detection_path: str):


        self.human_detection = False
        self.human_detection_path = human_detection_path
        self.human_detection_score = 0
        self.human_detection_data = [[], [], [], [], []]

        self.model = YOLO(human_detection_path)

        self.widget_frame = None
        self.human_detection_frame = None

        self.thread = Thread(target=self.run)
        self.stopped = False

    def start(self):
        self.thread.start()

    def run(self):
        while not self.stopped:
            if self.widget_frame:
                self.result = self.model.track(
                    self.widget_frame,
                    tracker="bytetrack.yaml",
                    imgsz=320,
                    classes=[0],
                    verbose=False,
                )
                # TODO: make this reset itself

                self.human_detection = True
                
    def stop(self):
        self.stopped = True

    def countIDs(self):
        counts = 0
        if self.result[0].boxes.id is not None:
            counts = max(self.result[0].boxes.id.int().cpu().tolist())
        return counts

    def getResultData(self):
        data = []

        if self.result[0].boxes.id is not None:
            for i, identity in enumerate(self.result[0].boxes.id):

                x, y, w, h = self.result[0].boxes.xywh[i]
                data.append([[int(identity)], [int(x)], [int(y)], [int(h)], [int(w)]])

        return data


class FaceWidget:

    def __init__(self,face_detection_path: str):
        self.face_detection = False
        self.face_detection_path = face_detection_path
        self.face_detection_score = 0
        self.face_detection_data = [[], [], [], [], []]

        self.model = YOLO(face_detection_path)

        self.widget_frame = None
        self.face_detection_frame = None

        self.thread = Thread(target=self.run)
        self.stopped = False

    def start(self):
        self.thread.start()

    def run(self):
        while not self.stopped:
            if self.widget_frame:
                self.result = self.model.track(
                    self.widget_frame,
                    tracker="bytetrack.yaml",
                    imgsz=320,
                    classes=[0],
                    verbose=False,
                )
                # TODO: make this reset itself
                self.face_detection = True

    def stop(self):
        self.stopped = True

    def count_ids(self):
        counts = 0
        if self.result[0].boxes.id is not None:
            counts = max(self.result[0].boxes.id.int().cpu().tolist())
        return counts

    def get_result_data(self)-> [Box]:
        data = []
        if self.result[0].boxes.id is not None:
            for i, identity in enumerate(self.result[0].boxes.id):
                x, y, w, h = self.result[0].boxes.xywh[i]

                data.append(Box(x,y,h,w))
        return data



class DeepFaceWidget:

    def __init__(self,database_path: str):

        self.database_path = database_path

        self.deepface_detections_data = [[], [], [], [], []]

        self.widget_frame = None

        self.thread = Thread(target=self.run)
        self.stopped = False

    def start(self):
        self.thread.start()

    def run(self):
        while not self.stopped:
            if self.widget_frame:
                self.result = DeepFace.find(
                    img_path=np.array(self.widget_frame),
                    db_path=self.database_path,
                    enforce_detection=False,
                    silent=True,
                    detector_backend="yolov8",
                    distance_metric="euclidean_l2",
                )

    def stop(self):
        self.stopped = True

    def get_result_data(self):
        data = []

        if len(self.result) > 0:
            for i, entry in enumerate(self.result):
                if not entry.empty:
                    identity = identity_from_string(entry["identity"][0])
                    x = entry["source_x"][0]
                    y = entry["source_y"][0]
                    w = entry["source_w"][0]
                    h = entry["source_h"][0]
                    data.append(Box([x], [y], [h], [w]))
        return data