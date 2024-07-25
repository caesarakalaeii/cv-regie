from abc import ABC
import os
import numpy as np
from logger import Logger
import cv2 as cv
from multiprocessing import Process
from shut_down_coordinator import Shutdown_Coordinator
import pyvirtualcam as pv

class OutputWiget(ABC):
    frame: np.ndarray
    sc: Shutdown_Coordinator

    def update_frame(self, frame: np.ndarray):
        if frame is None:
            return
        self.frame = frame.copy()

    def start(self):
        raise NotImplementedError(f'Class {self.__class__.__name__} does not implement start method')

    def stop(self):
        raise NotImplementedError(f'Class {self.__class__} should not be stopped.')

    def loop(self):
        raise NotImplementedError(f'Class {self.__class__.__name__} does not implement loop')

    def show_image(self):
        raise NotImplementedError(f'Class {self.__class__.__name__} does not support direct image access')


class ImageShowWidget(OutputWiget):
    "Wiget to show CameraFeed"

    frame: np.ndarray

    def __init__(
            self,
            window_title: str,
            l=Logger(),
            sc=Shutdown_Coordinator()
    ):
        self.sc = sc
        self.l = l
        self.stopped = True
        self.window_title = window_title
        self.frame = None
        self.process = None

    def start(self):
        self.stopped = False
        self.l.warning(
            'Starting ImageShowWidget with multiprocessing support')
        self.process = Process(target=self.run)
        self.process.start()

    def run(self):
        while not self.stopped and self.sc.running():
            self.show_image()

    def show_image(self):
        if self.stopped or not self.sc.running():
            self.stop()
            return
        if self.frame is None:
            return
        try:
            cv.imshow(self.window_title, self.frame)
            if cv.waitKey(1) == ord("q"):
                self.stop()
        except Exception as e:
            self.l.error(e)

    def stop(self):
        self.l.warning('Stopping ImageShowWidget')
        self.stopped = True
        self.sc.stop()
        cv.destroyAllWindows()
        if self.process:
            self.process.terminate()


class VirtualWebcamShowWidget(OutputWiget):
    "Wiget to show CameraFeed"

    frame: np.ndarray
    cam: pv.Camera

    def __init__(
            self,
            window_title: str,
            l=Logger(),
            sc=Shutdown_Coordinator(),
            output_size=(720, 1280, 30),
            convert_bgr_to_rgb=True
    ):
        self.sc = sc
        self.l = l
        self.stopped = True
        self.window_title = window_title
        self.height = output_size[0]
        self.width = output_size[1]
        self.fps = output_size[2]
        self.convert = convert_bgr_to_rgb
        self.frame = None
        self.process = None

    def update_frame(self, frame: np.ndarray):
        if frame is None:
            return
        if self.convert:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.frame = frame.copy()

    def start(self):
        self.stopped = False
        self.process = Process(target=self.run)
        self.process.start()

    def run(self):
        with pv.Camera(width=self.width, height=self.height, fps=self.fps, backend='obs' if os.name == 'nt' else 'v4l2loopback') as cam:
            while not self.stopped and self.sc.running():
                if self.frame is None:
                    continue
                cam.send(self.frame)
                cam.sleep_until_next_frame()

    def stop(self):
        self.l.warning('Stopping VirtualWebcamShowWidget')
        self.stopped = True
        self.sc.stop()
        if self.process:
            self.process.terminate()
