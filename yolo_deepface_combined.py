# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:11:02 2024

@author: Malte
"""
from typing import List

import cv2 as cv
import numpy as np
import json
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Results
from deepface import DeepFace
from pandas import DataFrame
import os


class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class TrackedPerson(object):
    name: str
    class_id: int
    confidence: float
    box: Box
    track_id: int

    def __init__(self, name, class_id, confidence, box, track_id):
        self.name = name
        self.class_id = class_id
        self.confidence = confidence
        self.box = box
        self.track_id = track_id

    @classmethod
    def from_json_string(cls, json_string: str):
        print(json_string)
        data = json.loads(json_string)[0]
        print(data)
        return cls(
            data["name"],
            data["class"],
            data["confidence"],
            Box(
                data["box"]["x1"],
                data["box"]["y1"],
                data["box"]["x2"],
                data["box"]["y2"],
            ),
            data["track_id"],
        )


class Frame(object):
    cv_frame: np.ndarray
    persons: List[TrackedPerson]
    faces: List[DataFrame]

    def __init__(self, cv_frame, persons, faces):
        self.cv_frame = cv_frame
        self.persons = persons
        self.faces = faces


ports = [0]
caps = []

for port in ports:
    caps.append(cv.VideoCapture(port))

for cap in caps:
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

yolo = YOLO("./models/detect/yolov8n.pt")

identities = {}

with os.scandir("./dev/database") as it:
    for entry in it:
        if not entry.name.startswith(".") and entry.is_dir():
            identities[entry.name] = []

print(f"Found {len(identities)} identities: {identities}")

while any(cap.isOpened() for cap in caps):
    for cap in caps:
        ret, frame = cap.read()

        if ret:
            result: [Results] = yolo.track(
                frame, tracker="botsort.yaml", classes=[0], imgsz=320, conf=0.5
            )

            human_frame = result[0].plot()
            tracked = []
            for res in result:
                tracked.append(TrackedPerson.from_json_string(res.tojson()))
            ids = []

            for d in tracked:
                print(d.name)
                ids.append(d.track_id)
            print(ids)
            dfs = DeepFace.find(
                img_path=np.array(human_frame),
                db_path="./dev/database",
                enforce_detection=False,
                silent=True,
            )
            test_frame = Frame(frame, tracked, dfs)
            for df in dfs:
                print(df)
            if not dfs[0].empty:
                print(f'{dfs[0]["identity"]} is {id}')

            cv.imshow("Example", human_frame)

    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()
