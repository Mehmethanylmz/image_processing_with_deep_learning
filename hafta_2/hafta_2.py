# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:53:13 2024

@author: Mete
"""
# Blurring
import cv2
import matplotlib.pyplot as plt
import numpy as np

import warnings 
warnings.filterwarnings("ignore")

img = cv2.imread("input_images/NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img),plt.axis("off"),plt.title("original"),plt.show()
dst2 = cv2.blur(img, ksize = (3,3))
plt.figure(), plt.imshow(dst2),plt.axis("off"),plt.title("ortalama blur")

gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX  = 7)
plt.figure(),plt.imshow(gb) ,plt.axis("off"),plt.title("Gauss Blur")

mb = cv2.medianBlur(img,ksize= 3)
plt.figure() , plt.imshow(mb),plt.axis("off"),plt.title("Medyan Blur")

def gaussianNoisy(image):
    row,col,ch = image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss 
    
    return noisy

img = cv2.imread("input_images/NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
plt.figure()
plt.imshow(img),plt.axis("off"),plt.title("original"),plt.show()

gaussianNoisyImage = gaussianNoisy(img)
plt.figure(),plt.imshow(gaussianNoisyImage) ,plt.axis("off"),plt.title("Gauss Noisy"),plt.show()


gb2 = cv2.GaussianBlur(gaussianNoisyImage, ksize = (3,3), sigmaX  = 7)
plt.figure(),plt.imshow(gb2) ,plt.axis("off"),plt.title("with Gauss Blur") 

img = cv2.imread("input_images/NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
plt.figure()
plt.imshow(img),plt.axis("off"),plt.title("original"),plt.show()

def saltPepperNoise(image):
    row,col,ch= image.shape
    s_vs_p= 0.5
    amount =0.004
    noisy = np.copy(image)
    
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1 , int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 1
    
    num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
    coords = [np.random.randint(0,i-1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy


    
spImage = saltPepperNoise(img)
plt.figure(),plt.imshow(spImage) ,plt.axis("off"),plt.title("SP Image"),plt.show()


mb2 = cv2.medianBlur(spImage.astype(np.float32),ksize= 3)
plt.figure() , plt.imshow(mb2),plt.axis("off"),plt.title("with Medyan Blur")



#%% Morphological

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("input_images/datai_team.jpg",0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off"), plt.title("Orijinal Img")

kernel = np.ones((5,5), dtype = np.uint8)
result = cv2.erode(img, kernel, iterations = 1)
plt.figure(), plt.imshow(result, cmap = "gray"), plt.axis("off"), plt.title("Erozyon")

result2 = cv2.dilate(img, kernel, iterations = 1)
plt.figure(), plt.imshow(result2, cmap = "gray"), plt.axis("off"), plt.title("Genisleme")

whiteNoise = np.random.randint(0,2, size = img.shape[:2])
whiteNoise = whiteNoise*255
plt.figure(), plt.imshow(whiteNoise, cmap = "gray"), plt.axis("off"), plt.title("White Noise")

noise_img = whiteNoise + img
plt.figure(), plt.imshow(noise_img, cmap = "gray"), plt.axis("off"), plt.title("Img w White Noise")

opening = cv2.morphologyEx(noise_img.astype(np.float32), cv2.MORPH_OPEN, kernel)
plt.figure(), plt.imshow(opening, cmap = "gray"), plt.axis("off"), plt.title("Acilma")

blackNoise = np.random.randint(0,2, size = img.shape[:2])
blackNoise = blackNoise*-255
plt.figure(), plt.imshow(blackNoise, cmap = "gray"), plt.axis("off"), plt.title("Black Noise")

black_noise_img = blackNoise + img 
black_noise_img[black_noise_img <= -245] = 0
plt.figure(), plt.imshow(black_noise_img, cmap = "gray"), plt.axis("off"), plt.title("Black Noise Img")

closing = cv2.morphologyEx(black_noise_img.astype(np.float32), cv2.MORPH_CLOSE, kernel)
plt.figure(), plt.imshow(opening, cmap = "gray"), plt.axis("off"), plt.title("Kapama")

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plt.figure(), plt.imshow(gradient, cmap = "gray"), plt.axis("off"), plt.title("Gradyan")
   
#%%  gradients

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("input_images/sudoku.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off"), plt.title("Orijinal Img")

sobelx = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 0, ksize = 5)
plt.figure(), plt.imshow(sobelx, cmap = "gray"), plt.axis("off"), plt.title("Sobel X")

sobely = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 0, dy =1, ksize = 5)
plt.figure(), plt.imshow(sobely, cmap = "gray"), plt.axis("off"), plt.title("Sobel Y")

laplacian = cv2.Laplacian(img, ddepth = cv2.CV_16S)
plt.figure(), plt.imshow(laplacian, cmap = "gray" ),plt.axis("off"), plt.title("Laplacian")
    
#%% Histogram

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("input_images/red_blue.jpg")
img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img_vis)

print(img.shape)

img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256])
print(img_hist.shape)
plt.figure(), plt.plot(img_hist)

color = ("b", "g", "r")
plt.figure()
for i, c in enumerate(color):
    hist = cv2.calcHist([img], channels = [i], mask = None, histSize = [256], ranges = [0,256])
    plt.plot(hist, color = c)
    
# 
golden_gate = cv2.imread("input_images/goldenGate.jpg")
golden_gate_vis = cv2.cvtColor(golden_gate, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(golden_gate_vis)    
    
print(golden_gate.shape)

mask = np.zeros(golden_gate.shape[:2], np.uint8)
plt.figure(), plt.imshow(mask, cmap = "gray")  

mask[1500:2000, 1000:2000] = 255
plt.figure(), plt.imshow(mask, cmap = "gray") 

masked_img_vis = cv2.bitwise_and(golden_gate_vis, golden_gate_vis, mask = mask)
plt.figure(), plt.imshow(masked_img_vis, cmap = "gray") 

masked_img = cv2.bitwise_and(golden_gate, golden_gate, mask = mask)
masked_img_hist = cv2.calcHist([golden_gate], channels = [0], mask = mask, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(masked_img_hist) 


img = cv2.imread("input_images/hist_equ.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray") 

img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(img_hist)

eq_hist = cv2.equalizeHist(img)
plt.figure(), plt.imshow(eq_hist, cmap = "gray") 

eq_img_hist = cv2.calcHist([eq_hist], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(eq_img_hist)
    
#%% Edge detection

import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("input_images/london.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

edges = cv2.Canny(image = img, threshold1 = 0, threshold2 = 255)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

med_val = np.median(img)
print(med_val)

low = int(max(0, (1 - 0.33)*med_val))
high = int(min(255, (1 + 0.33)*med_val))

print(low)
print(high)

edges = cv2.Canny(image = img, threshold1 = low, threshold2 = high)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

blurred_img = cv2.blur(img, ksize = (5,5))
plt.figure(), plt.imshow(blurred_img, cmap = "gray"), plt.axis("off")

med_val = np.median(blurred_img)
print(med_val)

low = int(max(0, (1 - 0.33)*med_val))
high = int(min(255, (1 + 0.33)*med_val))

print(low)
print(high)

edges = cv2.Canny(image = blurred_img, threshold1 = low, threshold2 = high)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

#%% Corner Detection

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("input_images/sudoku.jpg", 0)
img = np.float32(img)
print(img.shape)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")

dst = cv2.dilate(dst, None)
img[dst>0.2*dst.max()] = 1
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")


img = cv2.imread("input_images/sudoku.jpg", 0)
img = np.float32(img)
corners = cv2.goodFeaturesToTrack(img, 120, 0.01, 10)
corners = np.int64(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img, (x,y),3,(125,125,125),cv2.FILLED)
    
plt.imshow(img)
plt.axis("off")
    
#%% Contour Detection
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("input_images/contour.jpg",0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

# image, contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

external_contour = np.zeros(img.shape)
internal_contour = np.zeros(img.shape)

for i in range(len(contours)):
    if hierarch[0][i][3] == -1:
        cv2.drawContours(external_contour,contours, i, 255, -1)
    else: 
        cv2.drawContours(internal_contour,contours, i, 255, -1)

plt.figure(), plt.imshow(external_contour, cmap = "gray"),plt.axis("off")
plt.figure(), plt.imshow(internal_contour, cmap = "gray"),plt.axis("off")


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    