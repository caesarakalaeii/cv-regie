from camera_widgets import CameraWidget
from output_widgets import ImageShowWidget, OutputWiget
from utilities import calculate_frame_box_static, get_processed_frame
from threading import Thread
from logger import Logger
from shut_down_coordinator import Shutdown_Coordinator

class MODES:
    CV = 1
    VCAM = 2


class CV_Manager(object):
    
    def __init__(self, camera_ports = None,
                 output_mode = MODES.CV,
                 database_path = None, 
                 raw_resolution = (720, 1280),
                 raw_fps = 30,
                 target_resolution = (720, 1280),
                 human_detection_path: str = None,
                 face_detection_path: str = None,
                 debug:bool = False,
                 l:Logger = Logger(),
                 sc: Shutdown_Coordinator = Shutdown_Coordinator()
                ) -> None:
        self.camera_widgets = []
        self.ranking = []
        self.debug = debug
        self.debug_outputs = {}
        self.l = l
        self.sc = sc
        l.passingblue('Creating Manager')
        
        if debug:
            for cam in camera_ports:
                self.debug_outputs[cam] = []
        l.passingblue(f'Creating CameraWidgets {camera_ports}')
        for cam in camera_ports:
            camera = CameraWidget(cam, 
                             raw_resolution, 
                             raw_fps, 
                             human_detection_path, 
                             face_detection_path, 
                             database_path,
                             l, 
                             self.sc
                             )
            self.camera_widgets.append(
                camera
            )
            if debug:
                self.l.passing('Debug flag found, generating debug widgets')
                self.debug_outputs[cam].append(
                                                ImageShowWidget(f'Camera {cam} Raw',
                                                                l,
                                                                self.sc
                                                                )
                                               )
                                
            self.ranking.append(0)
        self.l.info(self.debug_outputs)
        if output_mode == MODES.CV:
            self.l.passing('Mode is CV, Creating Output Widget')
            self.output = ImageShowWidget('Output', l, self.sc)
        elif output_mode == MODES.VCAM:
            self.l.passing('Mode is VCAM, Creating Output Widget')
            raise NotImplementedError("Virtual Webcam not yet supported")
        else:
            raise ValueError("Mode not recognized")
        self.thread = Thread(target=self.loop)
        self.running = False
        
    def run(self):
        self.l.passingblue("Starting manager")
        
        if not self.running:
            self.running = True
            self.start_cam_widgets()
            if self.debug:
                debug:OutputWiget
                for debugs in self.debug_outputs.values():
                    for debug in debugs:
                        debug.start()
            self.output.start()
            self.thread.start()
        self.l.passing('Finished Setting up Manager')
        
    def update_cameras(self):
        cam_widget: CameraWidget
        for i, cam_widget in enumerate(self.camera_widgets):
            self.ranking[i]=cam_widget.get_ranking()
            if self.debug:
                debug:OutputWiget
                for debug in self.debug_outputs[cam_widget.port]:
                    debug.update_frame(cam_widget.frame)
        
    def get_cropped_frame_from_best_feed(self):
        self.l.info('Calculating rankings')
        best_feed = max(enumerate(self.ranking),key=lambda x: x[1])[0] #find index of highest ranking
        self.l.info(f'Best feed is feed{best_feed} with {self.ranking[best_feed]}')
        
        self.l.info('Processing frame')
        best_widget:CameraWidget = self.camera_widgets[best_feed]
        best_frame = best_widget.frame
        if best_frame is None or self.ranking[best_feed] == 0:
            return best_frame
        boxes = best_widget.get_detection_bounds()
        if boxes == []:
            self.l.warning('Boxes empty, continuing')
        else:
            box = calculate_frame_box_static(boxes)
        if box is None:
            return best_frame
        else:
            return get_processed_frame(box, best_widget.frame)
    
    def loop(self):
        try:
            while self.running:
                if not self.sc.running():
                    self.l.warning('Shutdown Detected exiting')
                    break
                self.update_cameras()
                cropped_frame = self.get_cropped_frame_from_best_feed()
                
                self.output.update_frame(cropped_frame)
        except Exception as e:
            self.l.error(e.with_traceback(e.__traceback__))
            self.running = False
            self.sc.stop()
            raise e
        self.sc.stop()
        exit(1)
            
            
    def start_image_show_no_threading(self):
        self.l.passingblue('Starting CV ImageShow')
        running = True
        # 'Start' outputs here for compatibility reasons
        for outputs in self.debug_outputs.values():
            output: ImageShowWidget
            for output in outputs:
                output.start()
        while running:
            if not self.sc.running():
                break
            for outputs in self.debug_outputs.values():
                output: ImageShowWidget
                for output in outputs:
                    output.show_image()
                    
            self.output.show_image()
            
                
    def start_cam_widgets(self):
        cam_widget: CameraWidget
        for cam_widget in self.camera_widgets:
            if cam_widget.stopped:
                self.l.info(f'Starting Camera {cam_widget.port}')
                cam_widget.start()