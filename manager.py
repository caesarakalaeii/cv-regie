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
from threading import Thread
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
        self.ranking = []
        self.debug = debug
        self.debug_outputs = {}
        self.l = l
        self.sc = sc
        self.output_mode = output_mode
        l.passingblue("Creating Manager")

        if debug:
            for cam_port in camera_ports:
                self.debug_outputs[cam_port] = []
        l.passingblue(f"Creating CameraWidgets {camera_ports}")
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
                self.debug_outputs[cam_port].append(
                    ImageShowWidget(f"Camera {cam_port} Raw", l, self.sc)
                )
                for widget in self.detection_widgets[cam_port]:
                    self.debug_outputs[cam_port].append(
                        ImageShowWidget(
                            f"Camera {cam_port} {widget.widget_type}", l, self.sc
                        )
                    )

            self.ranking.append(0)

        if output_mode == MODES.CV:
            self.l.passing("Mode is CV, Creating Output Widget")
            self.output = ImageShowWidget("Output", l, self.sc)
        elif output_mode == MODES.VCAM:
            self.l.passing("Mode is VCAM, Creating Output Widget")
            target_data = (target_resolution[0], target_resolution[1], raw_fps)
            self.output = VirtualWebcamShowWidget("Output", l, self.sc, target_data)
        else:
            raise ValueError("Mode not recognized")
        self.thread = Thread(target=self.run)
        self.running = False

    def start(self):
        self.l.passingblue("Starting manager")

        if not self.running:
            self.running = True
            self.start_cam_widgets()
            self.start_debug_widgets()
            self.init_detection_widgets()

            self.output.start()
            self.thread.start()

        self.l.passing("Finished Setting up Manager")

    def start_debug_widgets(self):
        if self.debug:
            debug: OutputWiget
            for debugs in self.debug_outputs.values():
                for debug in debugs:
                    debug.start()

    @timing
    def calc_rankings(self):
        for i, cam_widget in enumerate(self.camera_widgets):
            self.ranking[i] = self.get_ranking(cam_widget.port)

    @timing
    def get_ranking(self, cam_index: int):

        cam = self.get_cam_by_port(cam_index)
        if cam is None:
            return 0
        frame = cam.frame
        if frame is None:
            return 0
        persons = 0
        faces = 0
        for widget in self.detection_widgets[cam_index]:
            res = widget.count_ids()
            if widget.widget_type == "Human":
                persons = res
            if widget.widget_type == "Face":
                faces = res

        return calculate_ranking(frame.shape, persons, faces)

    def get_cam_by_port(self, cam_port):
        for cam in self.camera_widgets:
            if cam.port == cam_port:
                return cam
        return None

    @timing
    def get_detection_bounds(self, cam_index: int):
        return self.detection_widgets[cam_index][0].get_result_data()

    def annotate_frame(self, cam_index: int) -> np.ndarray:
        cam = self.get_cam_by_port(cam_index)
        if cam is None:
            return None
        frame = cam.frame
        if frame is None:
            return None
        return_frame = frame.copy()
        widget: DetectionWidget
        for widget in self.detection_widgets[cam_index]:
            return_frame = widget.plot_results(return_frame)

        return return_frame

    @timing
    def update_debug_outputs(self):
        cam_widget: CameraWidget
        if self.debug:
            for cam_index, outputs in enumerate(self.debug_outputs.values()):
                cam = self.camera_widgets[cam_index]
                debug: OutputWiget
                for i in range(len(outputs)):
                    if i == 0:
                        outputs[i].update_frame(
                            cam.frame
                        )  # first debug widget is raw image
                    else:
                        detection_widget = self.detection_widgets[cam.port][i - 1]
                        outputs[i].update_frame(detection_widget.plot_results())

    # @timing
    # def get_cropped_frame_from_best_feed(self):
    #     box = None
    #     self.l.info("Calculating rankings")
    #     self.calc_rankings()
    #     best_feed = np.argmax(self.ranking)
    #
    #     self.l.info(f"Best feed is feed{best_feed} with {self.ranking[best_feed]}")
    #     self.l.info("Processing frame")
    #     best_widget: CameraWidget = self.camera_widgets[best_feed]
    #     best_frame = best_widget.frame
    #     if best_frame is None or self.ranking[best_feed] == 0:
    #         return best_frame
    #     boxes = self.get_detection_bounds(best_widget.port)
    #     if boxes == []:
    #         self.l.warning("Boxes empty, continuing")
    #     else:
    #         box = calculate_frame_box_static(boxes)
    #     if box is None:
    #         return best_frame
    #     else:
    #         return get_processed_frame(box, best_widget.frame)

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

    def init_detection_widgets(self):
        widget: DetectionWidget
        for k, v in self.detection_widgets.items():
            for widget in v:
                if widget.stopped:
                    self.l.passing(
                        f"Starting DetectionWidget {widget.widget_type} for Camera {k}"
                    )
                    widget.start()

    @timing
    def update_widgets(self):
        widget: DetectionWidget
        for i, v in enumerate(self.detection_widgets.values()):
            for widget in v:
                widget.update_frame(self.camera_widgets[i].frame)

    def run(self):
        try:
            while self.running:
                if not self.sc.running():
                    self.l.warning("Shutdown Detected exiting")
                    break
                self.update_debug_outputs()
                self.update_widgets()
                cropped_frame = self.get_cropped_frame_from_best_feed()

                self.output.update_frame(cropped_frame)
        except Exception as e:
            self.l.error(e.with_traceback(e.__traceback__))
            self.running = False
            self.sc.stop()
            raise e
        self.sc.stop()
        exit()

    def start_image_show_no_threading(self):
        self.l.passingblue("Starting CV ImageShow")
        running = True
        # 'Start' outputs here for compatibility reasons
        for outputs in self.debug_outputs.values():
            for output in outputs:
                output.start()
        while running:
            if not self.sc.running():
                self.running = False
                break
            for outputs in self.debug_outputs.values():
                for output in outputs:
                    output.show_image()

            if self.output_mode == MODES.CV:
                self.output.show_image()

    def start_cam_widgets(self):
        cam_widget: CameraWidget
        for cam_widget in self.camera_widgets:
            if cam_widget.stopped:
                self.l.info(f"Starting Camera {cam_widget.port}")
                cam_widget.start()
