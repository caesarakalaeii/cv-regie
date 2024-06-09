from person_face_linker import LinkedFace, Person, Box
import numpy as np


class ProcessedFrame(object):
    
    persons: list[Person]
    frame: np.ndarray
    last_box: Box
    
    __frame_count:int
    __target_box:Box
    
    def __init__(self) -> None:
        self.persons = []
        self.frame = None
        self.last_box = None
        self.__frame_count = 0
        self.__target_box = None
        pass
    
    def add_person(self, person:Person) -> bool:
        if person in self.persons:
            return False
        self.persons.append(person)
        return True
    
    def remove_person(self, person:Person) -> bool:
        if person in self.persons:
            self.persons.remove(person)
            return True
        return False
    
    def find_person_by_id(self, id:int):
        for person in self.persons:
            if person.track_id == id:
                return person
        return None
            
    def remove_person_by_id(self, id:int) -> bool:
        
        person = self.find_person_by_id(id)
        if person:
            self.remove_person(person)
            return True
        return False
                
    def update_person(self, person:Person) -> None:
        
        self.remove_person_by_id(person.track_id)
        self.add_person(person)
        
    
    def update_frame(self, frame:np.ndarray) -> None:
        self.frame = frame
        
    def update_box(self, box:Box) -> None:
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
        return Box(lowest_x, lowest_y, highest_x, highest_y)
        
    def calculate_frame_box_rubber_band(self) -> Box:
        self.__target_box = self.calculate_frame_box_static()
        
        
        delta_x1  = self.__target_box.x1 - self.last_box.x1
        delta_y1  = self.__target_box.y1 - self.last_box.y1
        delta_x2  = self.__target_box.x2 - self.last_box.x2
        delta_y2  = self.__target_box.y2 - self.last_box.y2
        
        
        
        
    def get_processed_frame(self) -> np.ndarray:
        '''
        call after updates and box calculations, will return new frame
        '''
        return self.frame[self.last_box.y1:self.last_box.y2,self.last_box.x1:self.last_box.x2]  
        
                
        
        
