# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:34:09 2024

@author: Mete
"""

import cv2
import numpy as np


img = cv2.imread("input_images/lenna.png")
cv2.imshow("Original", img)

hor = np.hstack((img,img))
cv2.imshow("Horizontal",hor)

ver = np.vstack((img,img))
cv2.imshow("Dikey", ver)

cv2.waitKey(0)