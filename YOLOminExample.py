# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:11:02 2024

@author: Kilia
"""

import cv2 as cv
from ultralytics import YOLO

cap = cv.VideoCapture(1)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

model = YOLO('dev/models/face/yolov8n-face.pt')

while cap.isOpened():
    
    ret, frame = cap.read()
    
    if ret:
        
        result = model.track(frame,
                             tracker="botsort.yaml",
                             classes=[0],
                             imgsz=320,
                             conf=0.5)
        
        human_frame = result[0].plot()
        
        if result[0].boxes.id is not None:
            counts = max(result[0].boxes.id.int().cpu().tolist())

        cv.imshow('Example', human_frame)
    
    
    
    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows() 