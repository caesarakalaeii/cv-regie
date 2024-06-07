# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 08:00:12 2024

@author: Wittke
"""

import cv2 as cv
import numpy as np
from tensorflow import keras
from deepface import DeepFace

port = 5
resolution = [720, 1280]
camera_fps = 30

cap = cv.VideoCapture(port)
cap.set(cv.CAP_PROP_FRAME_WIDTH, resolution[1])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[0])
cap.set(cv.CAP_PROP_FPS, camera_fps)

#DeepFace.stream(db_path = "./database")

# objs = DeepFace.analyze(
#   img_path = "Z:/scripts/cv-regie/dev/database/Kilian/Kilian5.jpg", 
#   actions = ['age', 'gender', 'race', 'emotion'],
# )

# dfs = DeepFace.find(
#   img_path = "Z:/scripts/cv-regie/dev/database/Kilian/Kilian5.jpg",
#   db_path = "Z:/scripts/cv-regie/dev/database",
# )

while True:

    grabbed, frame = cap.read()
    if grabbed:

        dfs = DeepFace.find(
          img_path = np.array(frame),
          db_path = "./database",
          enforce_detection=False,
          silent=True
        )
        if not dfs[0].empty:
            print(dfs[0]["identity"][0])

        cv.imshow("Frame", frame)

    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()
cap.release()