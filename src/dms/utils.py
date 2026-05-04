# -*- coding: utf-8 -*-
# ---------------------

import numpy as np
from math import sin, cos, atan2, sqrt, pi
from threading import Timer
from collections import deque
from datetime import datetime, timedelta

class CircularBuffer(deque):
    # https://stackoverflow.com/questions/4151320/efficient-circular-buffer
    def __init__(self, size=0, fill_value=None):
        self.fill_value = fill_value
        if fill_value is not None:
            super(CircularBuffer, self).__init__(iterable=[fill_value for _ in range(size)], maxlen=size)
        else:
            super(CircularBuffer, self).__init__(maxlen=size)

    def reset(self):
        self.clear()
        if self.fill_value is not None:
            self.extend([self.fill_value for _ in range(self.__len__())])

    @property
    def average(self):
        return sum(self)/len(self)

class RepeatedTimer(object):
    # https://stackoverflow.com/questions/474528/what-is-the-best-way-to-repeatedly-execute-a-function-every-x-seconds
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()

def euler2mat(a,b,c, degrees=True):
    # type: (float, float, float, bool) -> np.ndarray

    if degrees:
        a, b, c = a * pi/180, b * pi/180, c * pi/180

    return np.array([[cos(c)*cos(b), -sin(c)*cos(a)+cos(c)*sin(b)*sin(a), sin(c)*sin(a)+cos(c)*sin(b)*cos(a)],
                     [sin(c)*cos(b), cos(c)*cos(a)+sin(c)*sin(b)*sin(a), -cos(c)*sin(a)+sin(c)*sin(b)*cos(a)],
                     [-sin(b), cos(b)*sin(a), cos(b)*cos(a)]])

def mat2euler(mat, degrees=True):
    # type: (np.ndarray, bool) -> (float, float, float)

    a = atan2(mat[2,1], mat[2,2])
    b = atan2(-mat[2,0], sqrt(mat[2,1]**2 + mat[2,2]**2))
    c = atan2(mat[1,0], mat[0,0])

    if degrees:
        a, b, c = a * 180/pi, b * 180/pi, c * 180/pi

    return a,b,c

def principal_roi(roi_list):
    #tl = (xmin, ymin)
    # br = (xmax, ymax)
    areas = [(r[2]-r[0])*(r[3]-r[1]) for r in roi_list]

    return areas.index(max(areas))

def eye_aspect_ratio(eye):
    # type: (np.array) -> float
    '''
    :param eye: Eye 6-points facial landmarks
    :return: Eye aspect ratio
    '''

    ear = (np.linalg.norm(eye[1] - eye[7]) + np.linalg.norm(eye[2] - eye[6]) + np.linalg.norm(eye[3] - eye[5])) \
          / (3*(np.linalg.norm(eye[0] - eye[4])))
    return ear

def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

if __name__ == '__main__':
    a,b,c,= 12.9, 65.2, 90.0

    mat = euler2mat(a,b,c)
    a1, b1, c1 = mat2euler(mat)

    print(a1,b1,c1)
