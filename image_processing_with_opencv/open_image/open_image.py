# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:31:24 2024

@author: Mete
"""
# Open image
import cv2

img = cv2.imread("input_images/messi5.jpg",0)
cv2.imshow("img",img)

k = cv2.waitKey(0) &0xFF

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('output_images/messibxcvinary.jpg', img)
    cv2.destroyAllWindows()
