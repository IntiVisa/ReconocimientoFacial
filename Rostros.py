# -*- coding: utf-8 -*-
"""
Created on Tue May 18 19:33:09 2021

@author: precalde
"""

import cv2
import numpy as np

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('rostros.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30,30),
	maxSize=(200,200))

for (x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('Lo que vemos',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


