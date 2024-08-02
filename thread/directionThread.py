# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:41:51 2024

@author: Wittke
"""

import time

import numpy as np

from threading import Thread

from thread.cameraThread import Frame


class DirectionWidget:
    "Director Widget which decides which camera is important"

    def __init__(
            self,
            frameObjects: [..., Frame],
            director_fartique: int,
            scoreFunction: [..., int] = [1, 2, 3]
            ):

        self.frameObjects = frameObjects
        self.director_fartique = director_fartique
        self.scoreFunction = scoreFunction
        self.bestPort = None
        self.bestIndex = None

        self.thread = Thread(target=self.run)
        self.stopped = False

    def start(self):

        self.thread.start()

    def run(self):
        while not self.stopped:
            time.sleep(self.director_fartique)
            score = np.array([0])
            for i, frameObject in enumerate(self.frameObjects):
                frameScore = self.calculateFrameScore(frameObject)

                if i == 0:
                    score = np.array([frameScore])
                else:
                    score = np.append(score, [frameScore])

            self.bestIndex = np.argmax(score)

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
