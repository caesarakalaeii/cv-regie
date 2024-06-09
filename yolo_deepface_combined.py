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
from person_face_linker import Person, LinkedFace
from pandas import DataFrame
import os


ports = [0]
caps = []


for port in ports:
    caps.append(cv.VideoCapture(port))

for cap in caps:
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

yolo = YOLO("./models/detect/yolov8n.pt")


linked_faces: [LinkedFace] = []
identities = {}
with os.scandir("./dev/database") as it:
    for entry in it:
        if not entry.name.startswith(".") and entry.is_dir():
            identities[entry.name] = []

print(f"Found {len(identities)} identities: {identities}")


while any(cap.isOpened() for cap in caps):
    for feed_id, cap in enumerate(caps):
        ret, frame = cap.read()

        if ret:
            result: [Results] = yolo.track(
                frame,
                tracker="botsort.yaml",
                classes=[0],
                imgsz=320,
                conf=0.5,
                persist=True,
            )

            human_frame = result[0].plot()
            tracked = []
            for res in result:
                person = Person.from_json_string(res.tojson(), frame, feed_id)
                if person is None:
                    continue
                tracked.append(person)

            if len(tracked) == 0:
                continue
            for person in tracked:
                person: Person

                found_face = False
                for face in linked_faces:
                    face: LinkedFace
                    found_face = face.register_person(person)
                    if found_face:
                        print(
                            f"Person with ID {person.track_id} on feed {feed_id} is linked, identity is {face.faceId}"
                        )
                if not found_face:
                    print(
                        f"Person with ID {person.track_id} on feed {feed_id} is not linked, creating a new link"
                    )
                    linked_faces.append(LinkedFace(person))

            cv.imshow("Example", human_frame)

    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()
