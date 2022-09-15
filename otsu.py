import cv2
import numpy as np;

im = cv2.imread("coins.bmp")
img = cv2.imread("coins.bmp", 0)
blurred  = cv2.GaussianBlur(img, (5, 5), 0)

atmc_img=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,5,2)
atgc_im=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,2)

th, im_th = cv2.threshold(atmc_img, 225, 255, cv2.THRESH_BINARY);
im_floodfill = im_th.copy()
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
im_out = im_th | im_floodfill_inv
 
kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(im_out, kernel, iterations = 2)
#dilation = cv2.dilate(erosion, kernel, iterations = 2)

bitwise_and = cv2.bitwise_and(im, im,mask=erosion)

#cv2.imshow("Floodfilled Image", im_floodfill)
#cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
#cv2.imshow("Foreground", im_out)
cv2.imshow('Result', erosion)
cv2.imshow('bitwise_and', bitwise_and)
#cv2.imshow("atmc_img", atmc_img)
#cv2.imshow("atgc_im", atgc_im)

cv2.waitKey()
cv2.destroyAllWindows()
#cv2.imwrite('bitwise_and.jpg', bitwise_and)