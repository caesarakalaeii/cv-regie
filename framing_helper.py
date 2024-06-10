from person_face_linker import LinkedFace, Person, Box
import numpy as np
import cv2 as cv


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


class ProcessedFrame(object):

    persons: list[Person]
    frame: np.ndarray
    last_box: Box

    __frame_count: int
    __target_box: Box

    def __init__(self) -> None:
        self.persons = []
        self.frame = None
        self.last_box = None
        self.__frame_count = 0
        self.__target_box = None
        pass

    def add_person(self, person: Person) -> bool:
        if person in self.persons:
            return False
        self.persons.append(person)
        return True

    def remove_person(self, person: Person) -> bool:
        if person in self.persons:
            self.persons.remove(person)
            return True
        return False

    def remove_persons_on_feed(self, feed_id: int) -> None:
        for person in self.persons:
            if feed_id == person.feed_id:
                self.remove_person(person)

    def find_person_by_id(self, id: int):
        for person in self.persons:
            if person.track_id == id:
                return person
        return None

    def remove_person_by_id(self, id: int) -> bool:

        person = self.find_person_by_id(id)
        if person:
            self.remove_person(person)
            return True
        return False

    def update_person(self, person: Person) -> None:

        self.remove_person_by_id(person.track_id)
        self.add_person(person)

    def update_frame(self, frame: np.ndarray) -> None:
        self.frame = frame

    def update_box(self, box: Box) -> None:
        self.last_box = box

    def calculate_frame_box_static(self) -> Box:
        lowest_x = 99999
        highest_x = 0
        lowest_y = 99999
        highest_y = 0

        for person in self.persons:
            if person.box.x1 < lowest_x:
                lowest_x = person.box.x1
            if person.box.x2 > highest_x:
                highest_x = person.box.x2
            if person.box.y1 < lowest_y:
                lowest_y = person.box.y1
            if person.box.y2 > highest_y:
                highest_y = person.box.y2

        box = Box(lowest_x, lowest_y, highest_x, highest_y)
        box = pad_to_16by9(box, target_shape=(16, 9))

        return box

    def calculate_frame_box_rubber_band(self) -> Box:
        self.__target_box = self.calculate_frame_box_static()

        delta_x1 = self.__target_box.x1 - self.last_box.x1
        delta_y1 = self.__target_box.y1 - self.last_box.y1
        delta_x2 = self.__target_box.x2 - self.last_box.x2
        delta_y2 = self.__target_box.y2 - self.last_box.y2

    def get_processed_frame(
        self, interpolation=cv.INTER_LANCZOS4, target_shape=(1280, 720)
    ) -> np.ndarray:
        """
        call after updates and box calculations, will return new frame
        """
        frame = self.frame[
            self.last_box.y1 : self.last_box.y2, self.last_box.x1 : self.last_box.x2
        ]

        return cv.resize(frame, target_shape, interpolation=interpolation)
