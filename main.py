

import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Set before any possible imports that use multiprocessing

from manager import CvManager, MODES
from shut_down_coordinator import Shutdown_Coordinator
from utilities import os_sensitive_backslashes, ensure_dir_exists
from logger import Logger
from multiprocessing import Process


def run_manager():
    ports = [0]
    resolution = [720, 1280]
    camera_fps = 20
    debug = False
    mode = MODES.CV

    human_detection_path = os_sensitive_backslashes("models/detection/yolov8n.pt")
    face_detection_path = os_sensitive_backslashes("models/face/yolov8n-face.pt")
    database_path = os_sensitive_backslashes("database")
    ensure_dir_exists(database_path)
    l = Logger(True)
    sc = Shutdown_Coordinator(l)
    manager = CvManager(
        ports,
        mode,
        database_path,
        resolution,
        camera_fps,
        resolution,
        human_detection_path,
        face_detection_path,
        debug=debug,
        l=l,
        sc=sc,
    )
    manager.start()

if __name__ == "__main__":
    p = Process(target=run_manager)
    p.start()
    p.join()
