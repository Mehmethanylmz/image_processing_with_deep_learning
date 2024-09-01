# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:32:26 2024

@author: Mete
"""

import cv2
import time

video_name ="input_video/MOT17-04-DPM.mp4"

cap = cv2.VideoCapture(video_name)
print("width",cap.get(3))
print("height",cap.get(4))

if cap.isOpened() == False:
    print("error")
    
while True:
    ret, frame = cap.read()
    if ret == True:
        time.sleep(0.01)
        cv2.imshow("Video", frame)
    else: break
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()