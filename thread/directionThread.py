# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:41:51 2024

@author: Wittke
"""

import time

import numpy as np

from threading import Thread

from utils import Frame

class DirectionWidget:
    "Director Widget which decides which camera is important"

    def __init__(
            self,
            frameObjects: [..., Frame],
            director_fatigue: int,
            scoreFunction: [..., int] = [1, 3, 6]
            ):

        self.frameObjects = frameObjects
        self.director_fatigue = director_fatigue
        self.scoreFunction = scoreFunction
        self.bestIndex = None

        self.thread = Thread(target=self.run)
        self.stopped = False

    def start(self):

        self.thread.start()

    def run(self):
        while not self.stopped:
            time.sleep(self.director_fatigue)
            score = np.array([0])
            for i, frameObject in enumerate(self.frameObjects):
                frameScore = self.calculateFrameScore(frameObject)
                print(f'Framescore {frameScore} on Port {i}')
                
                if i == 0:
                    score = np.array([frameScore])
                else:
                    score = np.append(score, [frameScore])
            #score = score[::-1] # TODO: Fix this properly, as this only works for two cams atm
            self.bestIndex = np.argmax(score)
            print(f'Best Index {self.bestIndex}')

    def stop(self):

        self.stopped = True

    def calculateFrameScore(
            self,
            frameObject: Frame
            ):
        score = 0
        score += self.scoreFunction[0]*frameObject.pose_detection_score
        score += self.scoreFunction[1]*frameObject.face_detection_score
        score += self.scoreFunction[2]*frameObject.identification_detection_score

        return score
