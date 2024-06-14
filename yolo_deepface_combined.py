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
from framing_helper import ProcessedFrame, pad_to_16by9
from ultralytics import YOLO
from ultralytics.engine.results import Results
from deepface import DeepFace
from person_face_linker import Person, LinkedFace
from ranking_helper import RankingHelper
from pandas import DataFrame
import os


ports = [0, 5, 7]
caps = []
processed_frames = []
ranks = []
available_frames = []
yolos = []
frame_counts = []


for port in ports:
    caps.append(cv.VideoCapture(port))

for cap in caps:
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    processed_frames.append(ProcessedFrame())
    ranks.append(0)
    available_frames.append(0)
    frame_counts.append(0)

    yolos.append(YOLO("./models/detect/yolov8n.pt"))
ranking_helper = RankingHelper((1280, 720))

linked_faces: list[LinkedFace] = []
identities = {}
unknowns = {}


with os.scandir("./dev/database") as it:
    for entry in it:
        if not entry.name.startswith(".") and entry.is_dir():
            identities[entry.name] = []
            linked_faces.append(LinkedFace(entry.name))

print(f"Found {len(identities)} identities: {identities}")


while any(cap.isOpened() for cap in caps):
    for feed_id, cap in enumerate(caps):
        ret, frame = cap.read()
        processed_frame: ProcessedFrame = processed_frames[feed_id]

        if ret:


            #find persons
            processed_frame.remove_persons_on_feed(
                feed_id
            )  # purge all persons, so no remains stick
            frame_counts[feed_id] += 1
            processed_frame.update_frame(frame)
            result: list[Results] = yolos[feed_id].track(
                frame,
                tracker="botsort.yaml",
                classes=[0],
                imgsz=320,
                conf=0.5,
                persist=True,
            )

            human_frame = result[0].plot()
            cv.imshow(f"Detect{feed_id}", human_frame)

            # reident Persons
            for res in result:
                for r in res:
                    print(r.tojson())

                    person = Person.from_json_string(r.tojson(), frame, feed_id)
                    if person is None:
                        continue
                    processed_frame.update_person(person)
            print(
                f"Found {len(processed_frame.persons)} persons: {processed_frame.persons}"
            )

            if len(processed_frame.persons) == 0:
                continue

            # Link Identity to persons

            for person in processed_frame.persons:
                person: Person
                found_face = False
                for face in linked_faces:
                    face: LinkedFace
                    if face.check_if_known_face(feed_id, person.track_id):
                        print(f"Found {person.track_id}, no df needed")
                        found_face = True
                    if (feed_id, person.track_id) in unknowns.keys():
                        if unknowns[(feed_id, person.track_id)] > 10:
                            print(
                                f"Person {person.track_id} is probably unknown will skip"
                            )
                            found_face = True  # Say we found a face after we unsuccessfully tried 10 times
                if found_face:
                    continue
                print(f"Did not find {person.track_id}, df needed")

                for face in linked_faces:
                    found_face = face.register_person(person)
                    if found_face:
                        print(
                            f"Person with ID {person.track_id} on feed {feed_id} is linked, identity is {face.face_id}"
                        )
                        continue
                    if (feed_id, person.track_id) in unknowns.keys():
                        unknowns[(feed_id, person.track_id)] = (
                            unknowns[(feed_id, person.track_id)] + 1
                        )
                    else:
                        unknowns[(feed_id, person.track_id)] = 1

            #Cut feed to size
            box = processed_frame.calculate_frame_box_static()
            processed_frame.update_box(box)
            new_frame = processed_frame.get_processed_frame()
            available_frames[feed_id] = new_frame

            # Ranking
            if frame_counts[feed_id] % 10 == 0 or frame_counts[feed_id] == 1:
                print(f"Frame count: {frame_counts[feed_id]}, New ranking")
                num_person = len(processed_frame.persons)
                num_faces, faces = ranking_helper.get_amount_amount_of_faces(
                    processed_frame.persons
                )
                ranking = ranking_helper.calculate_ranking(
                    new_frame.shape, num_person, num_faces
                )
                print(
                    f"Ranking of Feed {feed_id} is: {ranking}, found {num_faces} faces, {num_person} Persons, shape was {new_frame.shape}, following identities: {faces}"
                )
                ranks[feed_id] = ranking

        # Coohe best feed
        print(f"Rankings are: {ranks}")
        array = np.array([ranks])
        best_feed = np.argmax(array)
        cv.imshow("Example", available_frames[best_feed])

    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()
