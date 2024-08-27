# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 18:20:42 2024

@author: Mete
"""

import cv2
import numpy as np
from collections import deque

buffer_size = 10
pts = deque(maxlen=buffer_size)



blueLower = (84,98,0)
blueUpper = (179,255,255)

cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True: 
    success, imgOriginal = cap.read()
    if success:
        blurred =cv2.GaussianBlur(imgOriginal, (11,11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV IMAGE", hsv)
        #mavi için maskeleme
        
        mask = cv2.inRange(hsv, blueLower,blueUpper)
        cv2.imshow("mask Image", mask)
        #maske gürültü temizleme
        
        mask =cv2.erode(mask, None,iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow("Mask + erozyon ve genisleme", mask)
        
        #contur
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        
        if len(contours)>0:
            c = max(contours,key=cv2.contourArea)
            
            #etrafa diktörgen yerleştirme
            
            rect = cv2.minAreaRect(c)

            ((x,y),(width,height),rotation)= rect        
            s= "x: {},y : {},width : {}, height : {}, rotation : {},".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)
            
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            #moment
            M = cv2.moments(c)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            
            cv2.drawContours(imgOriginal, [box], 0,(0,255,255),2)
            cv2.circle(imgOriginal, center, 5, (255,0,255),-1)
            cv2.putText(imgOriginal, s, (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255),2) 
            
            
            
            
        pts.appendleft(center)
        
        for i in range (1, len(pts)):
            if pts[i-1] is None or pts[i] is None : continue
            cv2.line(imgOriginal, pts[i-1], pts[i] ,(0,255,0),3)
            
                
            
        cv2.imshow("Original Tespit", imgOriginal)
            
            
            
            
            
            
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break


#%%  Template Maching

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("input_images/cat.jpg",0)
print(img.shape)
template = cv2.imread("input_images/cat_face.jpg",0)
print(template.shape)
h,w = template.shape


methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']



for meth in methods:
    
    method = eval(meth) 
    res = cv2.matchTemplate(img, template, method)
    print(res.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    
    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap = "gray")
    plt.title("Eşleşen Sonuç"), plt.axis("off")
    plt.subplot(122), plt.imshow(img, cmap = "gray")
    plt.title("Tespit edilen Sonuç"), plt.axis("off")
    plt.suptitle(meth)
    plt.show()


#%% Feature Matching

import cv2
import matplotlib.pyplot as plt

chos = cv2.imread("input_images/chocolates.jpg", 0)
plt.figure(), plt.imshow(chos, cmap = "gray"),plt.axis("off")
cho = cv2.imread("input_images/nestle.jpg", 0)
plt.figure(), plt.imshow(cho, cmap = "gray"),plt.axis("off")

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(chos, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x: x.distance)


plt.figure()
img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags = 2)
plt.imshow(img_match), plt.axis("off"),plt.title("orb")


sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

kp1, des1 = sift.detectAndCompute(cho, None)
kp2, des2 = sift.detectAndCompute(chos, None)

matches = bf.knnMatch(des1, des2, k = 2)

guzel_eslesme = []
for match1, match2 in matches:
    if match1.distance < 0.75*match2.distance:
        guzel_eslesme.append([match1])
    
plt.figure()
sift_matches = cv2.drawMatchesKnn(cho,kp1,chos,kp2,guzel_eslesme,None, flags = 2)
plt.imshow(sift_matches), plt.axis("off"), plt.title("sift")



#%% Watershed

import cv2
import matplotlib.pyplot as plt
import numpy as np


coin = cv2.imread("input_images/coins.jpg")
plt.figure(), plt.imshow(coin), plt.axis("off")

coin_blur = cv2.medianBlur(coin, 13)
plt.figure(), plt.imshow(coin_blur), plt.axis("off")
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coin_gray, cmap="gray"), plt.axis("off")
ret, coin_thresh = cv2.threshold(coin_gray, 75, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")

#sürüm farkı
# _, contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coin, contours,i,(0,255,0),10)
plt.figure(),plt.imshow(coin),plt.axis("off")

coin = cv2.imread("input_images/coins.jpg")
plt.figure(), plt.imshow(coin), plt.axis("off")
coin_blur = cv2.medianBlur(coin, 13)
plt.figure(), plt.imshow(coin_blur), plt.axis("off")


coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coin_gray, cmap="gray"), plt.axis("off")

ret, coin_thresh = cv2.threshold(coin_gray, 65, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
plt.figure(), plt.imshow(opening, cmap="gray"), plt.axis("off")


dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
plt.figure(), plt.imshow(dist_transform, cmap="gray"), plt.axis("off")

ret, sure_foreground = cv2.threshold(dist_transform, 0.4*np.max(dist_transform),255,0)
plt.figure(), plt.imshow(sure_foreground, cmap="gray"), plt.axis("off")


sure_background = cv2.dilate(opening, kernel, iterations = 1)
sure_foreground = np.uint8(sure_foreground)
unknown = cv2.subtract(sure_background,sure_foreground)
plt.figure(), plt.imshow(unknown, cmap="gray"), plt.axis("off")


ret, marker = cv2.connectedComponents(sure_foreground)
marker = marker + 1
marker[unknown == 255] = 0
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")
marker = cv2.watershed(coin,marker)
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")

contours, hierarchy = cv2.findContours(marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coin, contours,i,(255,0,0),2)
plt.figure(),plt.imshow(coin),plt.axis("off")

#%% Face Detection


import cv2
import matplotlib.pyplot as plt


einstein = cv2.imread("input_images/einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")
face_cascade = cv2.CascadeClassifier("input_images/haarcascade_frontalface_default.xml")
face_rect = face_cascade.detectMultiScale(einstein)

for (x,y,w,h) in face_rect:
    cv2.rectangle(einstein, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")

barce = cv2.imread("input_images/barcelona.jpg", 0)
plt.figure(), plt.imshow(barce, cmap = "gray"), plt.axis("off")
face_rect = face_cascade.detectMultiScale(barce, minNeighbors = 7)

for (x,y,w,h) in face_rect:
    cv2.rectangle(barce, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(barce, cmap = "gray"), plt.axis("off")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)
        for (x,y,w,h) in face_rect:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(255,255,255),10)
        cv2.imshow("face detect", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release()
cv2.destroyAllWindows()


#%% Cat Face Detection

import cv2
import os

files = os.listdir()
print(files)
img_path_list = []
for f in files:
    if f.endswith(".jpg"): 
        img_path_list.append(f)
print(img_path_list)

for j in img_path_list:
    print(j)
    image = cv2.imread(j)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    rects = detector.detectMultiScale(gray, scaleFactor = 1.045, minNeighbors = 2)
    
    for (i, (x,y,w,h)) in enumerate(rects):
        cv2.rectangle(image, (x,y), (x+w, y+h),(0,255,255),2)
        cv2.putText(image, "Kedi {}".format(i+1), (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255),2)
    cv2.imshow(j, image)
    
    if cv2.waitKey(0) & 0xFF == ord("q"): continue

#%% Custom Cascade

"""
1) veri seti:
    n,p
2) cascade programı indir
3) cascade oluştur
4) cascade kullanarak tespit algoritması yaz
"""
import cv2
import os

path = "images"
imgWidth = 180
imgHeight = 120

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 180)

global countFolder
def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(path + str(countFolder)):
        countFolder += 1
    os.makedirs(path+str(countFolder))

saveDataFunc()
count = 0
countSave = 0

while True: 
    success, img = cap.read()
    if success:
        img = cv2.resize(img, (imgWidth, imgHeight))
        if count % 5 == 0:
            cv2.imwrite(path+str(countFolder)+"/"+str(countSave)+"_"+".png",img)
            countSave += 1
            print(countSave)
        count += 1
        cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release()
cv2.destroyAllWindows()




































