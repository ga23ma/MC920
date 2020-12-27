import cv2
import numpy as np
import sys
import argparse
import math 
parser = argparse.ArgumentParser()

parser.add_argument("-b", "--bitPlan", help="0 (default) - to use the first bit, 1 - to use the two first bits, 2- to use the three first bits", type=int)
parser.add_argument("-i", "--image", help="Image tobe decodified", required=True)
parser.add_argument("-d", "--destiny", help="Name of destiny file", required=True)
args = parser.parse_args()

BIT_PLAN = 0
if args.bitPlan is not None:
	BIT_PLAN = args.bitPlan


image = cv2.imread(args.image, 1)
if image is None:
	raise Exception("Invalid image")


# initializing string  

  
# printing original string  
res=''
msg =''
row = image.shape[0]
col = image.shape[1]
for i in range(row):
	if (len(res) == 8):
		if (int(res,2)== 255):
			break
		msg += (chr(int(res,2)))
		res =''
	
	for j in range(col):
		if (len(res) == 8):
			if (int(res,2)== 255):
				break
			msg += (chr(int(res,2)))
			res =''
		
		for k in range(3):	
			if (len(res) == 8):
				if (int(res,2)== 255):
					break
				msg += (chr(int(res,2)))
				res =''
			res += (str((int(image[i][j][k]) >> 0 & 1)))
			
			if (BIT_PLAN > 0):
				if (len(res) == 8):
					if (int(res,2)== 255):
						break
					msg += (chr(int(res,2)))
					res =''
				res += (str((int(image[i][j][k]) >> 1 & 1)))

			
			if (BIT_PLAN > 1):
				if (len(res) == 8):
					if (int(res,2)== 255):
						break
					msg += (chr(int(res,2)))
					res =''
				res += (str((int(image[i][j][k]) >> 2 & 1)))




arquivo = open(args.destiny, 'w')
arquivo.writelines(msg)
arquivo.close()

  	
# using join() + ord() + format() 
# Converting String to binary 
 
#hile (res != ''):
#c = res[0]
#	res = res[1:]
#	print(c)
# printing result  
#print("The string after binary conversion : " + str(res)) 
#print("The string after binary conversion : " + str((int(res) >> 7 & 1))) 
