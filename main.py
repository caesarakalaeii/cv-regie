"""
    This Program is used to ease the solo content creation of creators in workshop enviroments.
    Authored by Malte and Kilian
"""

from manager import CV_Manager, MODES
from shut_down_coordinator import Shutdown_Coordinator
from utilities import os_sensitive_backslashes, ensure_dir_exists
from logger import Logger
import os





if __name__ == '__main__':
   
    ports = [0,1]
    resolution = [720, 1280]
    camera_fps = 30

    human_detection_path = os_sensitive_backslashes("models/detection/yolov8n.pt")
    face_detection_path = os_sensitive_backslashes("models/face/yolov8n-face.pt")
    database_path = os_sensitive_backslashes("database")
    ensure_dir_exists(database_path)
    l = Logger(True)
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
        l = l,
        sc= Shutdown_Coordinator(l)
        )
    manager.run()
    manager.start_image_show_no_threading()
