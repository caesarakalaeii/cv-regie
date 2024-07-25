from abc import ABC
import os
import numpy as np
from logger import Logger
import cv2 as cv
from threading import Thread
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
        self.thread = Thread(target=self.run)

    def start(self):
        self.stopped = False
        self.l.warning(
            'OpenCV does not support threading. \nNo thread will be started, use show_image and update_frame instead')

    def run(self):
        self.l.passingblue("Starting output widget")

        while not self.stopped:
            if not self.sc.running():
                self.l.warning('Shutdown Detected')
                self.stop()
                break
            self.show_image()

    def show_image(self):
        if self.stopped:
            self.sc.stop()
            return
        if not self.sc.running():
            self.stop()
            return
        if self.frame is None:
            #self.l.info("no frame recieved")
            return
        try:
            cv.imshow(self.window_title, self.frame)
        except Exception as e:
            self.l.error(e)
            # print(f"Port:{widget.port} - fps:{widget.fps}")

        if cv.waitKey(1) == ord("q"):
            self.stop()

    def stop(self):
        self.l.warning('Stopping Output Widget')
        self.stopped = True
        self.sc.stop()
        cv.destroyAllWindows()


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
        self.thread = Thread(target=self.run)
        self.height = output_size[0]
        self.width = output_size[1]
        self.fps = output_size[2]
        self.convert = convert_bgr_to_rgb
        if os.name == 'nt':
            self.backend = 'obs'
        else:
            self.backend = 'v4l2loopback'
        self.frame = None

    def update_frame(self, frame: np.ndarray):
        if frame is None:
            return
        if self.convert:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.frame = frame.copy()

    def start(self):
        self.stopped = False
        self.thread.start()

    def run(self):
        self.l.passingblue("Starting VCam output widget")

        with pv.Camera(width=self.width,
                       height=self.height,
                       fps=self.fps,
                       backend=self.backend
                       ) as cam:
            while not self.stopped:
                if not self.sc.running():
                    self.l.warning('Shutdown Detected')
                    self.stop()
                    break
                if self.stopped:
                    self.sc.stop()
                    return
                if not self.sc.running():
                    self.stop()
                    return
                if self.frame is None:
                    #self.l.info("no frame recieved")
                    continue
                cam.send(self.frame)

    def stop(self):
        self.l.warning('Stopping Output Widget')
        self.stopped = True
        self.sc.stop()




if __name__ == '__main__':

    l = Logger(True)
    l.passingblue("Starting Minimum example, only used for debugging purposes")

    captures = []

    ports = [0, 8]
    min_ex = []

    for i, port in enumerate(ports):
        l.passing("Creating VidCaps")
        if os.name == 'nt':
            captures.append(cv.VideoCapture(port, cv.CAP_DSHOW))
        else:
            captures.append(cv.VideoCapture(port))
        min_ex.append(ImageShowWidget(f'Minimum example Cap {port}', l))
        min_ex[i].start()

    while True:
        for i, port in enumerate(ports):
            cap: cv.VideoCapture = captures[i]
            grabbed, frame = cap.read()

            if grabbed:
                min_ex[i].update_frame(frame)
                min_ex[i].show_image()
            else:
                l.warning("No frame returned")
