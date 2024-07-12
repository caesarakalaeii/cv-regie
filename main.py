"""
    This Program is used to ease the solo content creation of creators in workshop enviroments.
    Authored by Malte and Kilian
"""

from manager import CV_Manager, MODES
from utilities import os_sensitive_backslashes
from logger import Logger
import os

def ensure_dir_exists(directory):
    """
    Check if a given directory exists, and if not, creates it.

    Args:
        directory (str): The path of the directory to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")



if __name__ == '__main__':
   
    ports = [0,1]
    resolution = [720, 1280]
    camera_fps = 30

    human_detection_path = os_sensitive_backslashes("models/detection/yolov8n.pt")
    face_detection_path = os_sensitive_backslashes("models/face/yolov8n-face.pt")
    database_path = os_sensitive_backslashes("database")
    ensure_dir_exists(database_path)

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
    manager.start_image_show_no_threading()
