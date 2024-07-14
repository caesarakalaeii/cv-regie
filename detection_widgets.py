from abc import ABC
import os

from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
from threading import Thread
from utilities import Box, identity_from_string, os_sensitive_backslashes, plot_bounding_boxes
from logger import Logger
from output_widgets import ImageShowWidget
import cv2 as cv

from shut_down_coordinator import Shutdown_Coordinator


class DetectionWidget(ABC):
    '''
    Absract class to generalize detection widgets
    '''

    widget_frame: np.ndarray | None
    stopped: bool
    detection: bool
    widget_type = "Base"
    sc: Shutdown_Coordinator
    l: Logger
    frame_count: int
    detection_frequency: int

    def start(self):
        raise NotImplementedError()

    def stop(self):
        self.stopped = True
        self.sc.stop()
        self.l.warning(f'Stopping {self.widget_type}')
        exit()

    def count_ids(self):
        raise NotImplementedError()

    def get_result_data(self):
        raise NotImplementedError()

    def update_frame(self, frame):
        self.widget_frame = frame
        self.frame_count += 1

    def plot_results(self, frame=None) -> np.ndarray:
        if frame is None:
            return plot_bounding_boxes(self.widget_frame, self.get_result_data())
        return plot_bounding_boxes(frame, self.get_result_data())

    def run_detection(self):
        raise NotImplementedError()

    def running(self):
        if not self.sc.running():
            self.l.warning('Shutdown Detected exiting')
            self.stop()
        return not self.stopped


class HumanWidget(DetectionWidget):
    widget_type = "Human"

    def __init__(self, human_detection_path: str,
                 l=Logger(),
                 sc=Shutdown_Coordinator(),
                 detection_frequency=10):

        self.detection_frequency = detection_frequency
        self.sc = sc
        self.detection = False
        self.human_detection_path = human_detection_path
        self.human_detection_score = 0
        self.human_detection_data = [[], [], [], [], []]
        self.l = l
        self.l.info('Creating HumanWidget')

        self.model = YOLO(human_detection_path)
        self.result = None
        self.widget_frame = None
        self.human_detection_frame = None

        self.thread = Thread(target=self.run)
        self.stopped = True
        self.frame_count = 0

    def start(self):
        if self.stopped:
            self.l.info('Starting HumanWidget')
            self.stopped = False
            self.thread = Thread(target=self.run)
            self.thread.start()

    def run(self):
        try:
            while self.running():
                if self.widget_frame is None:
                    continue
                if self.frame_count % self.detection_frequency == 0:  # only run detection every 10th frame yourself
                    self.run_detection()
        except Exception as e:
            self.l.error(e.with_traceback(e.__traceback__))
            self.stop()
            raise e

    def run_detection(self):
        try:
            self.result = self.model.track(
                self.widget_frame,
                tracker="bytetrack.yaml",
                imgsz=320,
                classes=[0],
                verbose=False,
            )
            self.widget_frame = None
        except AttributeError as e:
            self.l.warning(f'Detection failed, continuing: {e}')

    def count_ids(self):
        counts = 0
        if self.result is None:
            return counts
        if self.result[0].boxes is None:
            return counts
        if self.result[0].boxes.id is not None:
            counts = max(self.result[0].boxes.id.int().cpu().tolist())
        return counts

    def get_result_data(self):
        data = []
        if self.result is None:
            return data
        if self.result[0].boxes.id is not None:
            for i, identity in enumerate(self.result[0].boxes.id):
                x1, y1, x2, y2 = self.result[0].boxes.xyxy[i]
                data.append(Box(x1, y1, x2, y2))

        return data

    def plot_results(self, frame=None) -> np.ndarray:
        if frame is not None:
            self.update_frame(frame)
            self.run_detection()
        if self.result is None:
            return self.widget_frame
        if self.result == []:
            return self.widget_frame
        if self.result[0] is None:
            return self.widget_frame

        return self.result[0].plot()


class FaceWidget(DetectionWidget):
    widget_type = "Face"
    widget_frame: np.ndarray

    def __init__(self, face_detection_path: str,
                 l=Logger(),
                 sc=Shutdown_Coordinator(),
                 detection_frequency=10):
        self.detection = False
        self.face_detection_path = face_detection_path
        self.face_detection_score = 0
        self.face_detection_data = [[], [], [], [], []]
        self.sc = sc
        self.l = l
        self.detection_frequency = detection_frequency
        self.l.info('Creating FaceWidget')
        self.result = None
        self.frame_count = 0
        self.model = YOLO(face_detection_path)

        self.widget_frame = None
        self.face_detection_frame = None

        self.thread = Thread(target=self.run)
        self.stopped = True

    def start(self):
        if self.stopped:
            self.l.info('Starting FaceWidget')
            self.stopped = False
            self.thread = Thread(target=self.run)
            self.thread.start()

    def run(self):
        try:
            while self.running():
                if self.widget_frame is None:
                    continue
                if self.frame_count % self.detection_frequency == 0:  # only run detection every 10th frame yourself
                    self.run_detection()
        except Exception as e:
            self.l.error(e.with_traceback(e.__traceback__))
            self.stop()
            raise e

    def run_detection(self):
        try:
            self.result = self.model.track(
                self.widget_frame,
                tracker="bytetrack.yaml",
                imgsz=320,
                classes=[0],
                verbose=False,
            )
            self.widget_frame = None

        except AttributeError as e:
            self.l.warning(f'Detection failed, continuing: {e}')

    def count_ids(self):
        counts = 0
        if self.result is None:
            return 0
        if self.result[0].boxes.id is not None:
            counts = max(self.result[0].boxes.id.int().cpu().tolist())
        return counts

    def get_result_data(self):
        data = []
        if self.result is None:
            return data
        if self.result[0].boxes.id is not None:
            for i, identity in enumerate(self.result[0].boxes.id):
                x1, y1, x2, y2 = self.result[0].boxes.xyxy[i]
                data.append(Box(x1, y1, x2, y2))
        return data

    def plot_results(self, frame=None) -> np.ndarray:
        if self.result is None:
            return self.widget_frame
        if self.result == []:
            return self.widget_frame
        if self.result[0] is None:
            return self.widget_frame

        if frame is not None:
            self.update_frame(frame)
            self.run_detection()
        return self.result[0].plot()


class DeepFaceWidget(DetectionWidget):
    widget_type = "DeepFace"
    widget_frame: np.ndarray

    def __init__(self, database_path: str,
                 l=Logger(),
                 sc=Shutdown_Coordinator(),
                 detection_frequency=10):

        self.database_path = database_path
        self.sc = sc
        self.deepface_detections_data = [[], [], [], [], []]
        self.result = None
        self.frame_count = 0
        self.detection_frequency = detection_frequency
        self.l = l
        self.l.info('Creating DeepFaceWidget')

        self.widget_frame = None

        self.thread = Thread(target=self.run)
        self.stopped = True

    def start(self):
        if self.stopped:
            self.l.info('Starting DeepFaceWidget')
            self.stopped = False
            self.thread = Thread(target=self.run)
            self.thread.start()

    def run(self):
        try:
            while self.running():
                if self.widget_frame is None:
                    continue
                if self.frame_count % self.detection_frequency == 0:  # only run detection every 10th frame yourself
                    self.run_detection()
        except Exception as e:
            self.l.error(e.with_traceback(e.__traceback__))
            self.stop()
            raise e

    def get_result_data(self):
        data = []
        if self.result is None:
            return data

        if len(self.result) > 0:
            for i, entry in enumerate(self.result):
                if not entry.empty:
                    identity = identity_from_string(entry["identity"][0])
                    x = entry["source_x"][0]
                    y = entry["source_y"][0]
                    w = entry["source_w"][0]
                    h = entry["source_h"][0]
                    data.append(Box(x, y, x + w, y + h, identity))
        return data

    def run_detection(self):
        try:
            self.result = DeepFace.find(
                img_path=np.array(self.widget_frame),
                db_path=self.database_path,
                enforce_detection=False,
                silent=True,
                detector_backend="yolov8",
                distance_metric="euclidean_l2"
            )
            self.widget_frame = None

        except AttributeError as e:
            self.l.warning(f'Detection failed, continuing: {e}')


if __name__ == '__main__':

    l = Logger(True)
    l.passingblue("Starting Minimum example, only used for debugging purposes")
    human_detection_path = os_sensitive_backslashes("models/detection/yolov8n.pt")
    face_detection_path = os_sensitive_backslashes("models/face/yolov8n-face.pt")
    database_path = os_sensitive_backslashes("database")

    captures = []

    ports = [0]
    min_ex = []
    outputs = []

    for i, port in enumerate(ports):
        l.passing("Creating VidCaps")
        if os.name == 'nt':
            captures.append(cv.VideoCapture(port, cv.CAP_DSHOW))
        else:
            captures.append(cv.VideoCapture(port))

        #Change this to test for different widgets
        widet_to_test = HumanWidget(human_detection_path, l)
        outputs.append(ImageShowWidget(f'Detection {widet_to_test.widget_type} Port {port}'))
        min_ex.append(widet_to_test)
        min_ex[i].start()
        outputs[i].start()
    try:
        while True:
            for i, port in enumerate(ports):
                cap: cv.VideoCapture = captures[i]
                grabbed, frame = cap.read()
                widget: DetectionWidget = min_ex[i]
                output: ImageShowWidget = outputs[i]
                if grabbed:
                    widget.update_frame(frame)
                    boxes = widget.get_result_data()
                    annotated_frame = widget.plot_results()
                    output.update_frame(annotated_frame)
                    output.show_image()
                    box: Box
                    for box in boxes:
                        l.info(box)
                else:
                    l.warning("No frame returned")
    except (KeyboardInterrupt, Exception) as e:
        l.error(f'{e}\nStopping widgets')
        for widget in min_ex:
            widget.stop()
        exit()
