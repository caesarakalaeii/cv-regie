from pandas import DataFrame
import numpy as np
import json
from deepface import DeepFace
from utilities import Box, identity_from_string


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


class LinkedFace(object):
    person_ids: dict

    face_id: int or None

    def __init__(self, identity: str = None):
        self.person_ids = {}
        self.face_id = identity

    def check_if_known_face(self, feed_id: int, personId: int) -> bool:
        if feed_id not in self.person_ids.keys():
            return False
        return personId == self.person_ids[feed_id]

    def register_person(self, person: Person):
        print(f"Registering person {person.track_id}")
        if person.feed_id in self.person_ids.keys():
            print(
                f"Person {person.track_id} already registered as {self.person_ids[person.feed_id]}"
            )
            if self.person_ids[person.feed_id] == person.track_id:
                print("they equal")
                return True  # track ID known, FaceID has been registered already

        print("New DF Scan")
        indentity = get_identity(person)
        if indentity == "":
            return False
        print(f"ident: {indentity}")
        if self.face_id is None:
            self.face_id = indentity  # if no faceID is known set it now

        print(f"Identity is: {indentity}")
        if indentity != self.face_id:
            return False  # Face doesn't match registered identity
        self.person_ids[person.feed_id] = (
            person.track_id
        )  # faceID has been set, register track id for feed
        return True
