# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:13:16 2024

@author: Mete
"""

import cv2
import os

directory = 'input_images'
files = os.listdir(directory)
print(files)
img_path_list = []
for f in files:
    if f.endswith(".jpg"): 
        img_path_list.append(os.path.join(directory, f))
print(img_path_list)

for j in img_path_list:
    print(j)
    image = cv2.imread(j)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.CascadeClassifier("input_images/haarcascade_frontalcatface.xml")
    rects = detector.detectMultiScale(gray, scaleFactor = 1.045, minNeighbors = 2)
    
    for (i, (x,y,w,h)) in enumerate(rects):
        cv2.rectangle(image, (x,y), (x+w, y+h),(0,255,255),2)
        cv2.putText(image, "Kedi {}".format(i+1), (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255),2)
    output_path = os.path.join("output_images", os.path.basename(j))
    cv2.imwrite(output_path, image)
    cv2.imshow(j, image)
    
    
    if cv2.waitKey(0) & 0xFF == ord("q"): continue