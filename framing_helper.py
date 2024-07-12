from person_face_linker import LinkedFace, Person
import numpy as np
import cv2 as cv
from utilities import pad_to_16by9, Box


class ProcessedFrame(object):
    
    '''
    DEPRECATED DO NOT USE, will be removed soon
    '''

    persons: list[Person]
    frame: np.ndarray
    last_box: Box

    def __init__(self) -> None:
        self.persons = []
        self.frame = None
        self.last_box = None
        raise DeprecationWarning('ProcessedFrame is deprecated and will be reemoved soon')

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

    

    

    

