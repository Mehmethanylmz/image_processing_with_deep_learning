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
#%%  Video Play
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
#%% Video Camera
import cv2


cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width,height)

writer =cv2.VideoWriter("output_video/Video_kaydi.mp4",cv2.VideoWriter_fourcc(*"DIVX"),20,(width,height))
while True:
    ret,frame = cap.read()
    cv2.imshow("Video", frame)
    
    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q") : break

cap.release()
writer.release()
cv2.destroyAllWindows()
#%% Resize Crop
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

#%% Shape and text

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


#%% Joining Images
import cv2
import numpy as np


img = cv2.imread("input_images/lenna.png")
cv2.imshow("Original", img)

hor = np.hstack((img,img))
cv2.imshow("Horizontal",hor)

ver = np.vstack((img,img))
cv2.imshow("Dikey", ver)

cv2.waitKey(0)

#%% Warp perspective
import cv2 
import numpy as np

img = cv2.imread("input_images/kart.png")
cv2.imshow("Original", img)

width = 400
height = 500

pts1 = np.float32([[205,1],[1,472],[540,150],[338,617]])
pts2 = np.float32([[0,0],[0,height],[width,0],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
print(matrix)

imgOutput = cv2.warpPerspective(img, matrix, (width,height))
cv2.imshow("Nihai Resim", imgOutput)


cv2.waitKey(0)

#%% Blending

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

#%% Image thresholding

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(img , cmap="gray")
plt.axis("off")
plt.show()

_, thresh_img =cv2.threshold(img, thresh= 60, maxval= 255, type = cv2.THRESH_BINARY)

plt.figure()
plt.imshow(thresh_img , cmap="gray")
plt.axis("off")
plt.show()


#uyarlamali esik degeri / adaptive

thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

plt.figure()
plt.imshow(thresh_img2 , cmap="gray")
plt.axis("off")
plt.show()













