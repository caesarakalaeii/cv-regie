# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:40:22 2024

@author: Wittke
"""

import cv2 as c
import os


# https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
def list_ports() -> tuple[list[int], list[int]]:
    """
    Test the ports and returns a tuple with the available ports
    and the ones that are working.
    """
    working_ports = []
    available_ports = []
    for dev_port in range(25):
        camera = c.VideoCapture(dev_port)
        if not camera.isOpened():
            print(f'Port {dev_port} not open')
        else:
            is_reading, _ = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print(f'Port {dev_port} is open and reads {w}x{h}')
                working_ports.append(dev_port)
            else:
                print(f'Port {dev_port} is open, but does not read {w}x{h}')
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports


if __name__ == '__main__':
    res = [1280, 720]
    fps = 30
    # timer in min
    timer = 1
    timer_frames = timer * 60 * fps
    # _, ports = list_ports()
    ports = [4,6,8]
    caps = []
    print(f'Available ports: {ports}\nStarting recording')
    for port in ports:
        caps.append(c.VideoCapture(port, c.CAP_V4L2))
    for port in ports:
        path = f'camera_{port}'
        if not os.path.exists(path):
            os.mkdir(path)

    for i, cap in enumerate(caps):
        cap.set(c.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(c.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(c.CAP_PROP_FPS, fps)
        print(f'Set resolution of camera {i} to {res[0]}x{res[1]}@{fps}')

    outputs = []
    for i, cap in enumerate(caps):
        output = c.VideoWriter(
            f'camera{i}/camera_{i}.avi',
            c.VideoWriter_fourcc(*'MJPG'), fps, (res[0], res[1]))
        outputs.append(output)

    isOpened = []
    for cap in caps:
        isOpened.append(cap.isOpened())
    if not all(isOpened):
        print("Cannot open camera")
        exit()
    first_frame = True
    atm_frame = 0
    while True:
        # Capture frame-by-frame
        cap_frame = []
        for cap in caps:
            _, frame = cap.read()
            cap_frame.append(frame)

        # Write the resulting frame
        for i, port in enumerate(ports):
            if first_frame:
                c.imwrite(f'camera_{i}/camera_{i}.png', cap_frame[i])
            outputs[i].write(cap_frame[i])
        first_frame = False
        # Display the resulting frame
        for i, cap in enumerate(caps):
            c.imshow(f'camera{i}', cap_frame[i])
        if c.waitKey(1) == ord('q'):
            break
        atm_frame += 1
        if atm_frame >= timer_frames:
            break
    # When everything done, release the capture
    for cap in caps:
        cap.release()
    c.destroyAllWindows()
