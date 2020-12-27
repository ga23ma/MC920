import cv2
import numpy as np 
image = cv2.imread("baboon_coded.png",1)
cv2.imshow("img",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
row = image.shape[0]
col = image.shape[1]
bitplan = (int('00000001', 2))
for i in range(row):
	for j in range(col):
		image[i][j][0] = (image[i][j][0] & bitplan)/bitplan * 255
		image[i][j][1] = (image[i][j][1] & bitplan)/bitplan * 255
		image[i][j][2] = (image[i][j][2] & bitplan)/bitplan * 255 
cv2.imshow("img",image)
cv2.imwrite("bitcut.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()