from widgets import CameraWidget
from threading import Thread

class CV_Manager(object):
    
    def __init__(self, camera_ports:[int] = None,
                 model_path = None, 
                 database_path = None, 
                 raw_resolution = (720, 1280),
                 raw_fps = 30,
                 target_resolution = (720, 1280),
                 human_detection_path: str = None,
                face_detection_path: str = None,
                 ) -> None:
        self.camera_widgets = []
        self.ranking = []
        for cam in camera_ports:
            self.camera_widgets.append(
                CameraWidget(cam, 
                             raw_resolution, 
                             raw_fps, 
                             human_detection_path, 
                             face_detection_path, 
                             database_path)
            )
            self.ranking.append(0)
            
        self.thread = Thread(target=self.loop)
        self.running = False
        
    def run(self):
        if not self.running:
            self.running = True
            self.thread.start()
            self.start_cam_widgets()
            
            
    def loop(self):
        while self.running:
            cam_widget: CameraWidget
            for i, cam_widget in enumerate(self.camera_widgets):
                self.ranking[i](cam_widget.get_ranking())
            best_feed = max(enumerate(self.ranking),key=lambda x: x[1])[0] #find index of highest ranking
            
            
                
    def start_cam_widgets(self):
        cam_widget: CameraWidget
        for cam_widget in self.camera_widgets:
            if not cam_widget.stopped:
                cam_widget.start()