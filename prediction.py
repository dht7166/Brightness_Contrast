from model import BRC
import cv2
import numpy as np

model = BRC()
model.model.load_weights('brc_2.h5')

image = cv2.imread('dark1.jpg')
image = cv2.resize(image,(256,256))
image = np.divide(image,255.0)
image = np.expand_dims(image,axis=0)
pred = model.model.predict(image)
alpha = pred[0][0][0][0]
beta = pred[0][0][0][1]
print(alpha,beta)
image = cv2.imread('dark1.jpg')
image = cv2.resize(image,(512,512))
pred = cv2.convertScaleAbs(image,alpha = alpha,beta = beta*50)
cv2.imshow('result',pred)
cv2.imshow('original',image)
cv2.waitKey(0)
cv2.imwrite('dark1-fixed.jpg',pred)