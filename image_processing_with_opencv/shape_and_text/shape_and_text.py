# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:33:45 2024

@author: Mete
"""

import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)
cv2.waitKey(0)


#(resim,başlangıç noktası , bitiş noktası , renk)
cv2.line(img, (100,100), (512,512), (0,255,0),3)


#(resim,baslangıç noktası , bitiş noktası , renk)
cv2.rectangle(img, (0,0),(256,256),(0,0,255),cv2.FILLED)
cv2.circle(img, (45,310), 45, (255,0,0),cv2.FILLED)


cv2.putText(img, "Resim", (0,400), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))


cv2.imshow("çizgili resim", img)
cv2.waitKey(0)