from abc import ABC


class OutputWiget(ABC):
    
    def update_frame(self, frame):
        raise NotImplementedError()
    
    def start(self):
        raise NotImplementedError()
    
    def stop(self):
        raise NotImplementedError()
        
    def loop(self):
        raise NotImplementedError()
    
class ImageShowWidget(OutputWiget):
    "Main widget to show all camera feeds"

    def __init__(
        self,
        port: int,
        resolution: [int, int],
        camera_fps: int,
        human_detection_path: str,
        face_detection_path: str,
        database_path: str,
    ):

        self.port = port
        self.resolution = resolution
        self.camera_fps = camera_fps

        self.human_detection_path = human_detection_path
        self.face_detection_path = face_detection_path
        self.database_path = database_path

        self.camera_widgets = []

        self.stopped = False

        self.start()

    def start(self):

        for port in self.ports:
            widget = CameraWidget(
                port,
                self.resolution,
                self.camera_fps,
                self.human_detection_path,
                self.face_detection_path,
                self.database_path,
            )
            self.camera_widgets.append(widget)

        for widget in self.camera_widgets:
            widget.start()

        self.run()

    def run(self):

        while not self.stopped:
            for widget in self.camera_widgets:
                if widget.grabbed:
                    frame = widget.frame
                    cv.imshow(f"Camera-{widget.port}-normal", frame)

                if widget.human_detection:
                    frame = widget.human_detection_frame
                    cv.imshow(f"Camera-{widget.port}-human", frame)

                if widget.face_detection:
                    frame = widget.face_detection_frame
                    cv.imshow(f"Camera-{widget.port}-face", frame)

                    # print(f"Port:{widget.port} - fps:{widget.fps}")

                if cv.waitKey(1) == ord("q"):
                    self.stop()

    def stop(self):
        self.stopped = True
        for widget in self.camera_widgets:
            widget.stop()
        cv.destroyAllWindows()
    