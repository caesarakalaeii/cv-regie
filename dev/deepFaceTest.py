# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 08:00:12 2024

@author: Wittke
"""

import cv2 as cv
import numpy as np
from tensorflow import keras
from deepface import DeepFace

from timeit import default_timer as timer

port = 1
resolution = [720, 1280]
camera_fps = 30

cap = cv.VideoCapture(port)
cap.set(cv.CAP_PROP_FRAME_WIDTH, resolution[1])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[0])
cap.set(cv.CAP_PROP_FPS, camera_fps)

fps = 0
db_path = "./database"

#DeepFace.stream(db_path = "./database")

# objs = DeepFace.analyze(
#   img_path = "Z:/scripts/cv-regie/dev/database/Kilian/Kilian5.jpg", 
#   actions = ['age', 'gender', 'race', 'emotion'],
# )

# dfs = DeepFace.find(
#   img_path = "Z:/scripts/cv-regie/dev/database/Kilian/Kilian5.jpg",
#   db_path = "Z:/scripts/cv-regie/dev/database",
# )
frame_counter = 0
time_start = timer()
idenify_counter = 0
idenify_percent = 0
idenify_counter_original = 0
while True:
    
    grabbed, frame = cap.read()
    if grabbed:
        dfs = DeepFace.find(
          img_path = np.array(frame),
          db_path = db_path,
          enforce_detection=False,
          silent=True,
          detector_backend = "yolov8"
        )
        if not dfs[0].empty:
            #print(dfs[0]["identity"][0][dfs[0]["identity"][0].index(db_path + "\\")+len(db_path + "\\"):][:dfs[0]["identity"][0][dfs[0]["identity"][0].index(db_path + "\\")+len("db_path + \\"):].index("\\")])
            idenify_counter += 1
            
        cv.imshow("Frame", frame)

        frame_counter += 1
        time_end = timer()
    
        if time_end - time_start > 1:
            fps = frame_counter
            idenify_percent = idenify_counter / fps
            idenify_counter_original = idenify_counter
            time_start = timer()
            frame_counter = 0
            idenify_counter = 0
        print(f"fps: {fps} - counter: {idenify_counter_original} - percent: {idenify_percent}")
    
    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()
cap.release()