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

model = YOLO("yolov8n-pose.pt")

while cap.isOpened():
    
    ret, frame = cap.read()
    
    if ret:
        
        result = model.track(frame,
                             tracker="botsort.yaml",
                             classes=[0],
                             imgsz=320,
                             conf=0.5)
        
        #human_frame = result[0].plot()
        keypoint = result[0].keypoints.xy[0]
        
        cv.circle(frame,(int(keypoint[0][0]), int(keypoint[0][1])), 10, (0,0,255), -1) #nose
        cv.circle(frame,(int(keypoint[1][0]), int(keypoint[1][1])), 10, (0,0,255), -1) #l eye
        cv.circle(frame,(int(keypoint[2][0]), int(keypoint[2][1])), 10, (0,0,255), -1) #r eye
        cv.circle(frame,(int(keypoint[3][0]), int(keypoint[3][1])), 10, (0,0,255), -1) #l ear
        cv.circle(frame,(int(keypoint[4][0]), int(keypoint[4][1])), 10, (0,0,255), -1) #r ear
        
        if result[0].boxes.id is not None:
            counts = max(result[0].boxes.id.int().cpu().tolist())

        cv.imshow('Example', frame)
    
    
    
    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows() 