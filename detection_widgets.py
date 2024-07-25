from abc import ABC
import numpy as np
from multiprocessing import Process
from utilities import Box, plot_bounding_boxes
from logger import Logger
import cv2 as cv
from shut_down_coordinator import Shutdown_Coordinator
from ultralytics import YOLO
from deepface import DeepFace


class DetectionWidget(ABC):
    """
    Abstract class for generalizing detection widgets.
    """
    widget_frame: np.ndarray
    stopped: bool
    detection: bool
    widget_type = "Base"
    sc: Shutdown_Coordinator
    l: Logger
    frame_count: int
    detection_frequency: int
    process: Process

    def __init__(self, l, sc):
        self.l = l
        self.sc = sc
        self.stopped = True
        self.process = None

    def start(self):
        if self.stopped:
            self.stopped = False
            self.process = Process(target=self.run)
            self.process.start()

    def stop(self):
        self.stopped = True
        if self.process:
            self.process.terminate()
        self.sc.stop()

    def run(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def update_frame(self, frame: np.ndarray):
        if frame is None:
            return
        self.widget_frame = frame.copy()
        self.frame_count += 1

    def plot_results(self, frame=None) -> np.ndarray:
        if frame is None:
            return plot_bounding_boxes(self.widget_frame, self.get_result_data())
        return plot_bounding_boxes(frame, self.get_result_data())

    def get_result_data(self):
        raise NotImplementedError("Subclasses must implement this method.")


class HumanWidget(DetectionWidget):
    widget_type = "Human"

    def __init__(self, human_detection_path, l, sc, detection_frequency=10):
        super().__init__(l, sc)
        self.model = YOLO(human_detection_path)
        self.detection_frequency = detection_frequency
        self.result = None

    def run(self):
        while not self.stopped:
            if self.sc.running() and self.widget_frame is not None and self.frame_count % self.detection_frequency == 0:
                self.run_detection()

    def run_detection(self):
        self.result = self.model(self.widget_frame)

    def get_result_data(self):
        boxes = []
        for *xyxy, conf, cls in self.result.xyxy[0]:
            if cls == 0:  # Assuming class 0 is human
                x1, y1, x2, y2 = map(int, xyxy)
                boxes.append(Box(x1, y1, x2, y2))
        return boxes


class FaceWidget(DetectionWidget):
    widget_type = "Face"

    def __init__(self, face_detection_path, l, sc, detection_frequency=10):
        super().__init__(l, sc)
        self.model = YOLO(face_detection_path)
        self.detection_frequency = detection_frequency
        self.result = None

    def run(self):
        while not self.stopped:
            if self.sc.running() and self.widget_frame is not None and self.frame_count % self.detection_frequency == 0:
                self.run_detection()

    def run_detection(self):
        self.result = self.model(self.widget_frame)

    def get_result_data(self):
        boxes = []
        for *xyxy, conf, cls in self.result.xyxy[0]:
            if cls == 1:  # Assuming class 1 is face
                x1, y1, x2, y2 = map(int, xyxy)
                boxes.append(Box(x1, y1, x2, y2))
        return boxes


class DeepFaceWidget(DetectionWidget):
    widget_type = "DeepFace"

    def __init__(self, database_path, l, sc, detection_frequency=10):
        super().__init__(l, sc)
        self.database_path = database_path
        self.detection_frequency = detection_frequency
        self.result = None

    def run(self):
        while not self.stopped:
            if self.sc.running() and self.widget_frame is not None and self.frame_count % self.detection_frequency == 0:
                self.run_detection()

    def run_detection(self):
        try:
            # Assuming widget_frame is converted to an image path or image array suitable for DeepFace
            self.result = DeepFace.analyze(img_path=self.widget_frame, actions=['emotion', 'age', 'gender', 'race'],
                                           enforce_detection=False)
        except Exception as e:
            self.l.error("DeepFace detection failed: {}".format(e))

    def get_result_data(self):
        if not self.result:
            return []
        data = [Box(x1=self.result['region']['x'], y1=self.result['region']['y'],
                    x2=self.result['region']['x'] + self.result['region']['w'],
                    y2=self.result['region']['y'] + self.result['region']['h'],
                    identifier=self.result['dominant_emotion'])]
        return data
