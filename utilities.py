"""
module comprised of utility functions, manly meant for static methods that provide rudimentary functions
"""

import os
from deepface import DeepFace
import numpy as np


class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.y2 = int(y2)
        self.x2 = int(x2)

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def cut_out(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y1 : self.y2, self.x1 : self.x2]


class Person(object):
    """
    Utility class to represent a person, only meant to extract YOLOS JSON response and used to pack feed id
    """

    name: str
    class_id: int
    confidence: float
    box: Box
    track_id: int
    feed_id: int
    human: np.ndarray

    def __init__(
        self, name, class_id, confidence, box, track_id, human: np.ndarray, feed_id
    ):
        self.name = name
        self.class_id = class_id
        self.confidence = confidence
        self.box = box
        self.track_id = track_id
        self.feed_id = feed_id
        self.human = human

    @classmethod
    def from_json_string(cls, json_string: str, frame: np.ndarray, feed_id: int):
        try:
            # print(json_string)

            data = json.loads(json_string)
            if len(data) == 0:
                return
            data = data[0]
            # print(data)
            box = Box(
                data["box"]["x1"],
                data["box"]["y1"],
                data["box"]["x2"],
                data["box"]["y2"],
            )
            return cls(
                data["name"],
                data["class"],
                data["confidence"],
                box,
                data["track_id"],
                box.cut_out(frame),
                feed_id,
            )
        except KeyError as e:
            print(f"Failed to parse json: {e}")


def get_identity(person: Person) -> str:
    """
    Runs DeepFace on given Person
    """

    dfs = DeepFace.find(
        img_path=np.array(person.human),
        db_path="./dev/database",
        enforce_detection=False,
        silent=True,
    )
    if dfs[0].empty:
        print("No known face detected")
        return ""  # no face found
    return identity_from_string(dfs[0]["identity"][0])


def identity_from_string(string: str):
    if os.name == "nt":
        return string.split("database\\")[1].split("\\")[0]
    return string.split("database/")[1].split("/")[0]


def pad_to_16by9(box: Box, target_shape=(16, 9)) -> Box:
    box_width = box.width()
    box_height = box.height()
    box_ratio = box_width / box_height
    target_ratio = target_shape[0] / target_shape[1]

    if box_ratio > target_ratio:
        # Current box is wider than target ratio, adjust height
        new_height = box_width / target_ratio
        delta_height = new_height - box_height
        new_y1 = box.y1 - delta_height / 2
        new_y2 = box.y2 + delta_height / 2
        # Ensure coordinates are non-negative
        if new_y1 < 0:
            delta_height = box.y1 - new_y1
            new_y1 = 0
            new_y2 = box_height + delta_height

        box.y1 = int(new_y1)
        box.y2 = int(new_y2)
    else:
        # Current box is taller than target ratio, adjust width
        new_width = box_height * target_ratio
        delta_width = new_width - box_width
        new_x1 = box.x1 - delta_width / 2
        new_x2 = box.x2 + delta_width / 2
        # Ensure coordinates are non-negative
        if new_x1 < 0:
            delta_width = box.x1 - new_x1
            new_x1 = 0
            new_x2 = box_width + delta_width
        box.x1 = int(new_x1)
        box.x2 = int(new_x2)

    return box


def calculate_ranking(
    frame_shape, person_count, face_count, max_res=(720, 1280)
) -> int:

    ranking = face_count * 3

    ranking += person_count * 2

    ranking += frame_shape[0] / max_res[0] + frame_shape[1] / max_res[1]

    return ranking


def get_amount_amount_of_faces(persons: list[Person]):
    faces = []
    if len(persons) == 0:
        return 0, faces
    for person in persons:
        identity = get_identity(person)
        if identity is "":
            continue
        if identity in faces:
            continue
        faces.append(identity)
    return len(faces), faces
