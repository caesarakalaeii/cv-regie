"""
module comprised of utility functions, manly meant for static methods that provide rudimentary functions
"""

import os
import time as tm
from datetime import time

import cv2 as cv
import numpy as np
from logger import Logger
from multiprocessing import Lock, Manager


manager = Manager()
counts = manager.dict()

def timing(func):
    global counts

    def wrapper(*args, **kwargs):
        global counts
        start = tm.time()
        result = func(*args, **kwargs)
        time_elapsed = tm.time() - start
        print(f"Function {func.__name__} took {time_elapsed * 1000} ms")
        with Lock():
            counts[func.__name__] = counts.get(func.__name__, 0) + 1
        return result

    return wrapper

class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    identifier: str

    def __init__(self, x1, y1, x2, y2, identifier: str = None):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.y2 = int(y2)
        self.x2 = int(x2)
        self.identifier = identifier

    def __str__(self) -> str:
        return f"X1:{self.x1} Y1:{self.y1} X2:{self.x2} Y2:{self.y2}"

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def cut_out(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y1: self.y2, self.x1: self.x2]

def plot_bounding_boxes(frame: np.ndarray, boxes) -> np.ndarray:
    for box in boxes:
        cv.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
    return frame

def identity_from_string(string: str):
    if os.name == "nt":
        return string.split("database\\")[1].split("\\")[0]
    return string.split("database/")[1].split("/")[0]

def os_sensitive_backslashes(string: str):
    if os.name == "nt":
        print(f"Detected Windows System, attempting conversion from path {string}")
        return string.replace("/", "\\")
    print(f"Detected Unix System, attempting conversion from path {string}")
    return string.replace("\\", "/")


def calculate_ranking(frame_shape, person_count, face_count, max_res=(720, 1280)) -> int:
    ranking = face_count * 3
    ranking += person_count * 2
    ranking += frame_shape[0] / max_res[0] + frame_shape[1] / max_res[1]
    return ranking

def ensure_dir_exists(directory):
    """
    Check if a given directory exists, and if not, creates it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")





@timing
def pad_to_target_shape(box: Box, target_shape=(16, 9)) -> Box:
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
            new_y1 = 0
            new_y2 = new_height
        new_x1, new_x2 = box.x1, box.x2
    else:
        # Current box is taller than target ratio, adjust width
        new_width = box_height * target_ratio
        delta_width = new_width - box_width
        new_x1 = box.x1 - delta_width / 2
        new_x2 = box.x2 + delta_width / 2
        # Ensure coordinates are non-negative
        if new_x1 < 0:
            new_x1 = 0
            new_x2 = new_width
        new_y1, new_y2 = box.y1, box.y2

    return Box(int(new_x1), int(new_y1), int(new_x2), int(new_y2))





@timing
def calculate_frame_box_static(boxes) -> Box:
    """
    return bounding box for all boxes in 16:9
    """

    lowest_x = 99999
    highest_x = 0
    lowest_y = 99999
    highest_y = 0

    for box in boxes:
        if box.x1 < lowest_x:
            lowest_x = box.x1
        if box.x2 > highest_x:
            highest_x = box.x2
        if box.y1 < lowest_y:
            lowest_y = box.y1
        if box.y2 > highest_y:
            highest_y = box.y2

    box = Box(lowest_x, lowest_y, highest_x, highest_y)
    box = pad_to_target_shape(box, target_shape=(16, 9))

    return box


@timing
def get_processed_frame(
        box: Box,
        frame: np.ndarray,
        interpolation=cv.INTER_LANCZOS4,
        target_shape=(1280, 720),
) -> np.ndarray:
    """
    call with bounding box to get the desired frame
    Make sure box is 16:9 using pad_to_16by9 or calculate_frame_box_static first
    """
    frame = frame[box.y1: box.y2, box.x1: box.x2]
    try:
        return cv.resize(frame, target_shape, interpolation=interpolation)
    except Exception as e:
        print(e)
        return frame


# https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
def list_ports(port_range: int, l: Logger):
    """
    Test the ports and returns a tuple with the available ports
    and the ones that are working.
    """
    working_ports = []
    available_ports = []
    for dev_port in range(port_range):
        if os.name == "nt":
            camera = cv.VideoCapture(dev_port, cv.CAP_DSHOW)
        else:
            camera = cv.VideoCapture(dev_port)
        if not camera.isOpened():
            l.fail(f"Port {dev_port} not open")
        else:
            is_reading, _ = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                l.passing(f"Port {dev_port} is open and reads {w}x{h}")
                working_ports.append(dev_port)
            else:
                l.warning(f"Port {dev_port} is open, but does not read {w}x{h}")
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports

"""
Module comprised of utility functions, mainly meant for static methods that provide rudimentary functions
"""





if __name__ == "__main__":
    l = Logger(True)
    port_range = 10
    l.warning("Starting search for open Ports \nThis might take a while!")
    available_ports, working_ports = list_ports(port_range, l)
    l.passingblue(f"Found Ports: {available_ports}\nWorking Ports are: {working_ports}")


if __name__ == "__main__":
    l = Logger(True)
    port_range = 10

    l.warning("Starting search for open Ports \nThis might take a while!")
    available_ports, working_ports = list_ports(port_range, l)
    l.passingblue(f"Found Ports: {available_ports}\nWorking Ports are: {working_ports}")
