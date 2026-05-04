# -*- coding: utf-8 -*-
# ---------------------

from math import ceil
import numpy as np

import cv2

def resize_with_pad(image, height=1080, width=1920):
    # type: (np.array, int, int) -> np.array
    """
    Resize and image adding a zero padding to keep aspect ratio
    :param image: numpy array (3,H,W) usually obtained by OpenCV
    :param height: desires output height
    :param width: desired output width
    :return: numpy array (3,height, width)
    """

    #target padding
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape

    #set width
    if h/height < w/width:
        new_height = ceil(h * width/w)
        image = cv2.resize(image, (width, new_height))

        # pad the height
        delta_h = height - new_height
        top = delta_h // 2
        bottom = delta_h - top

    #set height
    else:
        new_width = ceil(w * height/h)
        image = cv2.resize(image, (new_width, height))

        # pad the height
        delta_w = width - new_width
        left = delta_w // 2
        right = delta_w - left

    BLACK = [0, 0, 0]
    #apply padding
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    return image