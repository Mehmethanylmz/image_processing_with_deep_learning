# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 10:15:11 2024

@author: Mete
"""
import cv2
import os
from os.path import isfile, join

pathIn = r"input_data/img1"
pathOut = "output_video/MOT17-13-SDP.mp4"

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn,f))]
fps = 25
size = (1920,1080)
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"MP4V"), fps, size, True)

for i in files:
    print(i)
    filename = pathIn + "\\" + i
    img = cv2.imread(filename)
    out.write(img)
out.release()





















