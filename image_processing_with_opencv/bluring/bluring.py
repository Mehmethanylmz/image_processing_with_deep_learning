# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:00:00 2024

@author: Mete
"""

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