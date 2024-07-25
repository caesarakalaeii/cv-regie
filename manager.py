import numpy as np
import camera_widgets
from camera_widgets import CameraWidget
from detection_widgets import DetectionWidget, HumanWidget, FaceWidget, DeepFaceWidget
from output_widgets import ImageShowWidget, OutputWiget, VirtualWebcamShowWidget
from utilities import (
    calculate_frame_box_static,
    get_processed_frame,
    calculate_ranking,
    timing,
)
from multiprocessing import Process, Manager
from logger import Logger
from shut_down_coordinator import Shutdown_Coordinator

class MODES:
    CV = 1
    VCAM = 2

class CvManager(object):
    def __init__(
        self,
        camera_ports=None,
        output_mode=MODES.CV,
        database_path=None,
        raw_resolution=(720, 1280),
        raw_fps=30,
        target_resolution=(720, 1280),
        human_detection_path: str = None,
        face_detection_path: str = None,
        debug: bool = False,
        l: Logger = Logger(),
        sc: Shutdown_Coordinator = Shutdown_Coordinator(),
    ) -> None:
        self.camera_widgets = []
        self.detection_widgets = {}
        self.debug = debug
        self.debug_outputs = {}
        self.l = l
        self.sc = sc
        self.output_mode = output_mode
        l.passingblue("Creating Manager")

        for cam_port in camera_ports:
            camera = CameraWidget(
                cam_port,
                raw_resolution,
                raw_fps,
                l,
                self.sc,
            )
            self.camera_widgets.append(camera)
            human_detection_widget = HumanWidget(human_detection_path, l, self.sc)
            face_detection_widget = FaceWidget(face_detection_path, l, self.sc)
            deepface_detection_widget = DeepFaceWidget(database_path, l, self.sc)
            cam_widgets = [
                human_detection_widget,
                face_detection_widget,
                deepface_detection_widget,
            ]
            self.detection_widgets[cam_port] = cam_widgets
            if debug:
                self.l.passing("Debug flag found, generating debug widgets")
                self.debug_outputs[cam_port] = [
                    ImageShowWidget(f"Camera {cam_port} Raw", l, self.sc)
                ]
                for widget in cam_widgets:
                    self.debug_outputs[cam_port].append(
                        ImageShowWidget(
                            f"Camera {cam_port} {widget.widget_type}", l, self.sc
                        )
                    )

        if output_mode == MODES.CV:
            self.output = ImageShowWidget("Output", l, self.sc)
        elif output_mode == MODES.VCAM:
            target_data = (target_resolution[0], target_resolution[1], raw_fps)
            self.output = VirtualWebcamShowWidget("Output", l, self.sc, target_data)

        self.processes = []
        self.manager = Manager()
        self.ranking = self.manager.list([0] * len(camera_ports))

    def start(self):
        self.l.passingblue("Starting manager")
        for cam_widget in self.camera_widgets:
            cam_widget.start()

        if self.debug:
            for cam_port, debugs in self.debug_outputs.items():
                for debug in debugs:
                    debug.start()

        for widgets in self.detection_widgets.values():
            for widget in widgets:
                widget.start()

        self.output.start()
        main_process = Process(target=self.run)
        self.processes.append(main_process)
        main_process.start()

    def run(self):
        try:
            while self.sc.running():
                self.update_widgets()
                cropped_frame = self.get_cropped_frame_from_best_feed()
                self.output.update_frame(cropped_frame)
        except Exception as e:
            self.l.error(e.with_traceback(e.__traceback__))
            self.sc.stop()

    def update_widgets(self):
        for widgets in self.detection_widgets.values():
            for widget in widgets:
                widget.update_frame(self.get_frame_from_camera(widget.camera_port))

    def get_frame_from_camera(self, cam_port):
        for cam in self.camera_widgets:
            if cam.port == cam_port:
                return cam.frame
        return None

    @timing
    def get_cropped_frame_from_best_feed(self):
        self.l.info("Calculating rankings")
        self.calc_rankings()
        best_feed = np.argmax(self.ranking)

        best_ranking = self.ranking[best_feed]
        self.l.info(f"Best feed is feed{best_feed} with {best_ranking}")

        if best_ranking == 0:
            self.l.info("Best ranking is zero, skipping processing")
            return None

        best_widget = self.camera_widgets[best_feed]
        best_frame = best_widget.frame.copy()
        if best_frame is None:
            self.l.info("Best frame is None, skipping processing")
            return None

        self.l.info("Processing frame")
        boxes = self.get_detection_bounds(best_widget.port)
        if not boxes:
            self.l.warning("Boxes empty, returning best frame")
            return best_frame

        box = calculate_frame_box_static(boxes)
        if box is None:
            self.l.info("Calculated box is None, returning best frame")
            return best_frame

        return get_processed_frame(box, best_frame)

    @timing
    def calc_rankings(self):
        for i, cam_widget in enumerate(self.camera_widgets):
            self.ranking[i] = self.get_ranking(cam_widget.port)
