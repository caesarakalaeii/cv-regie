from pandas import DataFrame
import numpy as np
import json
from deepface import DeepFace


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


class LinkedFace(object):
    personIds: dict

    faceId: int or None

    def __init__(self, person: Person):
        self.personIds = {}
        self.faceId = None
        self.register_person(person)

    def register_person(self, person: Person):
        if person.feed_id in self.personIds:
            if self.personIds[person.feed_id] == person.track_id:
                return True  # track ID known, FaceID has been registered already
        dfs = DeepFace.find(
            img_path=np.array(person.human),
            db_path="./dev/database",
            enforce_detection=False,
            silent=True,
        )
        # TODO: Test for unkown faces, maybe generate db on the fly
        if dfs[0].empty:
            return False  # no face found

        if self.faceId is None:
            self.faceId = dfs[0]["identity"]  # if no faceID is known set it now

        print(f'Identity is: {dfs[0]["identity"]}')
        if not dfs[0]["identity"].equals(self.faceId):
            return False  # Face doesn't match registered identity
        self.personIds[person.feed_id] = (
            person.track_id
        )  # faceID has been set, register track id for feed
        return True
