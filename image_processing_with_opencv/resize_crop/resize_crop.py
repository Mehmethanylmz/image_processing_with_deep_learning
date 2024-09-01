# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:33:26 2024

@author: Mete
"""

import cv2

img = cv2.imread("input_images/lenna.png")
print("Resim boyutu : " ,img.shape)
cv2.imshow("Original",img)

imgResized = cv2.resize(img, (300,300))
print("Resized Img Shape" ,imgResized.shape)
cv2.imshow("img",imgResized)


imgCropped = img[:200,:300]
cv2.imshow("kirpikresim", imgCropped)
cv2.waitKey(0)
