from abc import ABC

from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
from threading import Thread
from utilities import Box, identity_from_string, os_sensitive_backslashes
from logger import Logger
import cv2 as cv

class DetectionWidget (ABC):
    '''
    Absract class to generalize detection widgets
    '''
    
    widget_frame: np.ndarray
    stopped: bool
    detection: bool
    widget_type = "Base"
    
    def start(self):
        raise NotImplementedError()
    
    def stop(self):
        raise NotImplementedError()
    
    def count_ids(self):
        raise NotImplementedError()

    def get_result_data(self):
        raise NotImplementedError()
    
    def update_frame(self, frame):
        self.widget_frame = frame



class HumanWidget(DetectionWidget):
    
    widget_type = "Human"
    
    

    def __init__(self, human_detection_path: str, l= Logger()):


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
        
    
    def start(self):
        if self.stopped:
            self.l.info('Starting HumanWidget')
            self.stopped = False
            self.thread = Thread(target=self.run)
            self.thread.start()

    def run(self):
        try:
            while not self.stopped:
                if self.widget_frame is None:
                    continue
                self.result = self.model.track(
                    self.widget_frame,
                    tracker="bytetrack.yaml",
                    imgsz=320,
                    classes=[0],
                    verbose=False,
                )
                if len(self.result) != 0:
                    self.detection = True
                else:
                    self.detection = False
                self.widget_frame = None
        except Exception as e:
            self.l.error(e.with_traceback(e.__traceback__))
            self.stop()
            raise e
                
    def stop(self):
        self.stopped = True
        self.l.warning(f'Stopping {self.widget_type}')
        

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

                x, y, w, h = self.result[0].boxes.xywh[i]
                data.append(Box(x,y,x+w,y+h))

        return data
    
    def update_frame(self, frame):
        self.widget_frame = frame


class FaceWidget:
    
    widget_type = "Face"
    widget_frame: np.ndarray
    
    

    def __init__(self,face_detection_path: str, l = Logger()):
        self.detection = False
        self.face_detection_path = face_detection_path
        self.face_detection_score = 0
        self.face_detection_data = [[], [], [], [], []]
        
        self.l = l
        self.l.info('Creating FaceWidget')
        self.result = None

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
            while not self.stopped:
                if self.widget_frame is None:
                    continue
                self.result = self.model.track(
                    self.widget_frame,
                    tracker="bytetrack.yaml",
                    imgsz=320,
                    classes=[0],
                    verbose=False,
                )
                if len(self.result) != 0:
                    self.detection = True
                else:
                    self.detection = False
                self.widget_frame = None
        except Exception as e:
            self.l.error(e.with_traceback(e.__traceback__))
            self.stop()
            raise e
                

    def stop(self):
        self.stopped = True
        self.l.warning(f'Stopping {self.widget_type}')
        

    def count_ids(self):
        counts = 0
        if self.result is None:
            return 0
        if self.result[0].boxes.id is not None:
            counts = max(self.result[0].boxes.id.int().cpu().tolist())
        return counts

    def get_result_data(self)-> list[Box]:
        data = []
        if self.result is None:
            return data
        if self.result[0].boxes.id is not None:
            for i, identity in enumerate(self.result[0].boxes.id):
                x, y, w, h = self.result[0].boxes.xywh[i]

                data.append(Box(x,y,x+w,y+h))
        return data
    
    def update_frame(self, frame):
        self.widget_frame = frame



class DeepFaceWidget:
    
    widget_type = "DeepFace"
    widget_frame: np.ndarray
    
    

    def __init__(self,database_path: str, l= Logger()):

        self.database_path = database_path

        self.deepface_detections_data = [[], [], [], [], []]
        self.result = None
        
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
            while not self.stopped:
                if self.widget_frame is None:
                    continue
                self.result = DeepFace.find(
                    img_path=np.array(self.widget_frame),
                    db_path=self.database_path,
                    enforce_detection=False,
                    silent=True,
                    detector_backend="yolov8",
                    distance_metric="euclidean_l2",
                )
                self.widget_frame = None
        except Exception as e:
            self.l.error(e.with_traceback(e.__traceback__))
            self.stop()
            raise e

    def stop(self):
        self.stopped = True
        self.l.warning(f'Stopping {self.widget_type}')

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
                    data.append(Box(x,y,x+w,y+h))
        return data
    
    def update_frame(self, frame):
        self.widget_frame = frame
        

if __name__ == '__main__':
    
    l = Logger(True)
    l.passingblue("Starting Minimum example, only used for debugging purposes")
    human_detection_path = os_sensitive_backslashes("models/detection/yolov8n.pt")

    captures = []
    
    ports = [0]
    min_ex = []
    
    for i, port in enumerate(ports):
        l.passing("Creating VidCaps")
        captures.append(cv.VideoCapture(port))
        
        #Change this to test for different widgets
        widet_to_test = HumanWidget(human_detection_path, l)
        
        min_ex.append(widet_to_test)
        min_ex[i].start()
    
    while True:
        for i, port in enumerate(ports):
            cap: cv.VideoCapture = captures[i]
            grabbed, frame = cap.read()
            widget: DetectionWidget = min_ex[i]
            
            if grabbed:
                widget.update_frame(frame)
                l.info(widget.get_result_data())
            else:
                l.warning("No frame returned")