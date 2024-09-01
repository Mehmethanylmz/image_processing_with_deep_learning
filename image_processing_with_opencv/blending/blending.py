# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:34:58 2024

@author: Mete
"""

import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("input_images/img1.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("input_images/img2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)


img1 = cv2.resize(img1, (600,600))
print(img1.shape)
img2 = cv2.resize(img2, (600,600))
print(img2.shape)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)


blended = cv2.addWeighted(src1= img1, alpha= 0.1, src2 = img2, beta = 0.3, gamma = 0)

plt.figure()
plt.imshow(blended)