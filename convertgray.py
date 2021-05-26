import numpy as np
import cv2

img = cv2.imread("a3.png")
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 160, 255, 0)
cv2.imwrite("a3gray.png", thresh)