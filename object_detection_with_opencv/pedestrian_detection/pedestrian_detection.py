# -*- coding: utf-8 -*-
import cv2
import os



output_directory = 'output_images'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

directory = 'input_images'
files = os.listdir(directory)
img_path_list = []

for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(os.path.join(directory, f))
print(img_path_list)


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in img_path_list:
    print(imagePath)
    image = cv2.imread(imagePath)
    (rects, weights) = hog.detectMultiScale(image, padding = (8,8), scale = 1.05)
    for (x,y,w,h) in rects:
        cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,255),2)
    output_path = os.path.join(output_directory, os.path.basename(imagePath))
    cv2.imwrite(output_path, image)     
    cv2.imshow("Yaya: ",image)
    if cv2.waitKey(0) & 0xFF == ord("q"): continue
    