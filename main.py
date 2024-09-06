# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 08:51:32 2024

@author: Wittke
"""

from thread.leadThread import LeadWidget

if __name__ == "__main__":
    ports = [0, 1]
    resolution = [720, 1280]
    camera_fps = 30
    skipped_frames = [2, 10] #[yolo,deepface]
    director_fatigue = 3 #secounds between bestFrame shifts
    verbose = False #shows fps
    picturesque = 2 # 1 -> camera frame, 2 -> + yolo frame

    pose_detection_path = "model/pose/yolov8n-pose.pt"
    database_path = "./database"

    imageShow = LeadWidget(
        ports,
        resolution,
        camera_fps,
        skipped_frames,
        director_fatigue,
        pose_detection_path,
        database_path,
        verbose,
        picturesque
    )
    imageShow.start()
