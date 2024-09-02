# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:35:23 2024

@author: Mete
"""

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("input_images/img1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(img , cmap="gray")
plt.axis("off")
plt.show()

_, thresh_img =cv2.threshold(img, thresh= 60, maxval= 255, type = cv2.THRESH_BINARY)

plt.figure()
plt.imshow(thresh_img , cmap="gray")
plt.axis("off")
plt.show()


#uyarlamali esik degeri / adaptive

thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

plt.figure()
plt.imshow(thresh_img2 , cmap="gray")
plt.axis("off")
plt.show()
