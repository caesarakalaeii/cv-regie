# -*- coding: utf-8 -*-
"""
Created on Thu May  2 08:46:22 2024

@author: Wittke
"""

from threading import Thread
import cv2 as cv
import numpy as np
import time


class CameraWidget(object):
    """ doc """

    def __init__(self, cam_port: int = 0,
                 res: [int, int] = [720/2, 1280/2],
                 fps: int = 30,
                 eachother_fps: int = 3) -> None:
        self.frame_count = 0
        self.gray_image = None
        self.all_widgets = None
        self.eachother_fps = eachother_fps
        self.show_t = None
        self.run_t = None
        self.stopped = False
        self.grabbed = None
        self.frame = None
        self.port = cam_port
        self.fps = fps
        self.run_t = Thread(target=self.run)
        self.faces_t = Thread(target=self.find_and_mark_faces)
        self.humans_t = Thread(target=self.find_and_mark_humans)
        self.hog = cv.HOGDescriptor()
        self.hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
        # Maybe self.show_t = Thread(target=self.show_frame)
        self.cap = cv.VideoCapture(cam_port, cv.CAP_V4L2)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, res[1])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, res[0])
        self.cap.set(cv.CAP_PROP_FPS, fps)
        self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def set_all_widgets(self, all_widgets) -> None:
        self.all_widgets = all_widgets

    def start(self) -> None:
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.run_t.start()
        self.faces_t.start()
        self.humans_t.start()
        #self.show_t.start()

        #    print('Huh')

    def run(self) -> None:
        prev_frame_time = time.time()
        while not self.stopped:
            self.gray_image = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
            new_frame_time = time.time()

            # Calculating the fps

            # fps will be number of frame processed in given time frame
            # since their will be most of time error of 0.001 second
            # we will be subtracting it to get more accurate result
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # converting the fps into integer
            self.fps = int(fps)

            # converting the fps to string so that we can display it on frame
            # by using putText function
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.cap.read()
                self.frame_count = self.frame_count + 1
                # time.sleep(1 / self.fps)

    def find_and_mark_faces(self):
        old_faces = self.face_cascade.detectMultiScale(self.gray_image,
                                                       scaleFactor=1.1,
                                                       minNeighbors=5,
                                                       minSize=(30, 30))
        no_faces = 0
        while not self.stopped:
            # Detect faces in the image
            if self.frame_count % self.eachother_fps == 0:
                faces = self.face_cascade.detectMultiScale(self.gray_image,
                                                           scaleFactor=1.1,
                                                           minNeighbors=5,
                                                           minSize=(30, 30))
                if len(faces) > 0:
                    old_faces=faces
                else:
                    no_faces += 1
                    if no_faces >20:
                        old_faces = faces
                # Draw rectangles around the detected faces
            for x, y, w, h in old_faces:
                cv.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def find_and_mark_humans_old(self):

        old_humans, _ = self.hog.detectMultiScale(self.gray_image)
        no_humans = 0
        while not self.stopped:
            if self.frame_count % self.eachother_fps == 0:
                humans, _ = self.hog.detectMultiScale(self.gray_image)
                if len(humans) > 0:
                    old_humans=humans
                else:
                    no_humans += 1
                    if no_humans >20:
                        old_humans = humans
            has_init = True

            # loop over all detected humans
            for (x, y, w, h) in old_humans:
                pad_w, pad_h = int(0.15 * w), int(0.01 * h)
                cv.rectangle(self.frame, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)

    def find_and_mark_humans(self):
        net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layers_names = net.getLayerNames()
        output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        while True:
            # Get frame dimensions
            height, width, channels = self.frame.shape

            # Preprocess the frame for YOLO
            blob = cv.dnn.blobFromImage(self.frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            # Loop over each detection and filter out humans
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5 and class_id == 0:  # 0 corresponds to 'person' class in YOLO
                        # Object detected is a person
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Non-max suppression to remove overlapping boxes
            indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Enumerate humans and draw bounding boxes
            humans = []
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    cv.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    humans.append((x, y, x + w, y + h))


    def stop(self) -> None:
        self.stopped = True
        self.cap.release()
        cv.destroyAllWindows()
        exit(69)

    def show_frame(self) -> None:

        # while not self.stopped:
            if self.grabbed:
                # Our operations on the frame come here

                # font which we will be using to display FPS
                font = cv.FONT_HERSHEY_SIMPLEX
                # time when we finish processing for this frame


                # putting the FPS count on the frame
                cv.putText(self.frame, f'{self.fps}', (7, 70), font, 3, (100, 255, 0), 3, cv.LINE_AA)
                cv.imshow(f'camera{self.port}', self.frame)
                if cv.waitKey(1) == ord("q"):
                    for w in self.all_widgets:
                        w.stop()


if __name__ == '__main__':

    ports = [4,6,8]
    widgets = []

    for port in ports:
        widgets.append(CameraWidget(cam_port=port, eachother_fps=1))

    for widget in widgets:
        widget.set_all_widgets(widgets)
        widget.start()

    while True:
        for widget in widgets:
            widget.show_frame()
            if widget.stopped:
                exit(69)
