"""
    This Program is used to ease the solo content creation of creators in workshop enviroments.
    Authored by Malte and Kilian
"""

from manager import CV_Manager, MODES
from logger import Logger




if __name__ == '__main__':
   
    ports = [0,1]
    resolution = [720, 1280]
    camera_fps = 30

    human_detection_path = "models/detection/yolov8n.pt"
    face_detection_path = "models/face/yolov8n-face.pt"
    database_path = "./database"

    manager = CV_Manager(
        ports,
        MODES.CV,
        database_path,
        resolution,
        camera_fps,
        resolution,
        human_detection_path,
        face_detection_path,
        debug=True,
        l = Logger(True)
        )
    manager.run()
