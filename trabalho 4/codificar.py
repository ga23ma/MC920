import cv2
import numpy as np
import sys
import argparse
import math 
parser = argparse.ArgumentParser()

parser.add_argument("-b", "--bitPlan", help="0 (default) - to use the first bit, 1 - to use the two first bits, 2- to use the three first bits", type=int)
parser.add_argument("-i", "--image", help="Image to codify", required=True)
parser.add_argument("-m", "--message", help="Message to be codified", required=True)
parser.add_argument("-d", "--destiny", help="Name of destiny file", required=True)
args = parser.parse_args()

BIT_PLAN = 0
if args.bitPlan is not None:
	BIT_PLAN = args.bitPlan

f = open(args.message, 'r')

image = cv2.imread(args.image, 1)
if image is None:
	raise Exception("Invalid image")

end = chr(255)
# initializing string  
test_str = f.read()
test_str += end
# printing original string  
res = ''.join(format(ord(i), '08b') for i in test_str)

row = image.shape[0]
col = image.shape[1]
for i in range(row):
	if(res == ''):
		break
	for j in range(col):
		if(res == ''):
			break
		for k in range(3):	
			if(res == ''):
				break
			c = int (res[0]) | 254
			image[i][j][k] = (image[i][j][k]|1) & int(c)
			res = res[1:]
			#print(bin(image[i][j][k]))
			if(res == ''):
				break
			if (BIT_PLAN > 0):
				c = int (res[0])<<1 | 253
				image[i][j][k] = (image[i][j][k]|2) & int(c)
				res = res[1:]

			if(res == ''):
				break
			if (BIT_PLAN > 1):
				c = int (res[0])<<2 | 251
				image[i][j][k] = (image[i][j][k]|4) & int(c)
				res = res[1:]


cv2.imwrite(args.destiny, image)

  	

