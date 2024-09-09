# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 18:58:36 2024

@author: Mete
"""

import cv2

OPENCV_OBJECT_TRACKERS  = {
    "BOOSTING": cv2.legacy.TrackerBoosting_create,
    "MIL": cv2.legacy.TrackerMIL_create,
    "KCF": cv2.legacy.TrackerKCF_create,
    "TLD": cv2.legacy.TrackerTLD_create,
    "MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create,
    "MOSSE": cv2.legacy.TrackerMOSSE_create,
    "CSRT": cv2.legacy.TrackerCSRT_create
}

tracker_name = "MIL"

trackers = cv2.legacy.MultiTracker_create()

video_path = "MOT17-04-DPM.mp4"
cap = cv2.VideoCapture(video_path)

fps = 25     
f = 0
while True:
    
    ret, frame = cap.read()
    (H, W) = frame.shape[:2]
    frame = cv2.resize(frame, dsize = (960, 540))
    
    (success , boxes) = trackers.update(frame)
    
    info = [("Tracker", tracker_name),
        	("Success", "Yes" if success else "No")]
    
    string_text = ""
    
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        string_text = string_text + text + " "
    
    cv2.putText(frame, string_text, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("t"):
        
        box = cv2.selectROI("Frame", frame, fromCenter=False)
    
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        trackers.add(tracker, frame, box)
    elif key == ord("q"):break

    f = f + 1
    
cap.release()
cv2.destroyAllWindows() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    