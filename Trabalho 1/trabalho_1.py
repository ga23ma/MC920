import cv2
import numpy as np
import sys
import argparse
import math 
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--filter", help="Filter number (1 to h1, 2 to h2, 3 to h3, 4 to h4, 5 to h5, 6 to h6, 7 to h7, 8 to h8, 9 to h9, 10 to h10, 11 to h11, 12 to h3+h4)", type=int, required=True)
parser.add_argument("-i", "--image", help="Image to apply the filter", required=True)
parser.add_argument("-d", "--destiny", help="Name of destiny file", required=True)
args = parser.parse_args()

def convolve (image, filt, factor):
	img = image.copy()
	backRow = round((len(filt)-1)/2)
	backCol = round((len(filt[0])-1)/2)
	row = image.shape[0]
	col = image.shape[1]
	for i in range(row):
		for j in range(col):
			intesity = 0
			for k in range(i-backRow, i+backRow+1):
				for l in range(j-backCol ,j+backCol+1):
					if k < 0 or l < 0 or k > row-1 or l > col-1:
						intesity += 0
						
					else:
						intesity += filt[k-(i-backRow)][l-(j-backCol)]* img[k][l][0]
			intesity = round(intesity/factor)			
			if intesity > 255:
				intesity = 255
			if intesity < 0:
				intesity = 0
			image[i][j] = intesity


	cv2.imshow('image',image)
	cv2.imwrite('nova.png', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows() #simple implementation

def convolve2 (image, filt, factor, dest):
	img = image.copy()
	backRow = round((len(filt)-1)/2)
	backCol = round((len(filt[0])-1)/2)

	padded_image = np.pad(img, [
        (backRow, backRow),
        (backCol, backCol),
        (0, 0)
    ])

	
	row = image.shape[0]
	col = image.shape[1]
	for x in range(backRow, row+backRow):
		for y in range(backCol, col+backCol):
			
			window = padded_image[(x - backRow):(x + 1 + backRow), (y - backCol):(y + 1 + backCol), 0]

			intesity = np.sum(filt * window, axis=(0, 1)) 
			
			intesity = round(intesity/factor)			
			if intesity > 255:
				intesity = 255
			if intesity < 0:
				intesity = 0
			image[x - backRow][y - backCol] = intesity
	cv2.imshow('image',image)
	cv2.imwrite(dest, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows() #optimazed using numpy

def convolveSum (image, filt, filt2, factor, dest):
	img = image.copy()
	backRow = round((len(filt)-1)/2)
	backCol = round((len(filt[0])-1)/2)

	padded_image = np.pad(img, [
        (backRow, backRow),
        (backCol, backCol),
        (0, 0)
    ])

	
	row = image.shape[0]
	col = image.shape[1]
	for x in range(backRow, row+backRow):
		for y in range(backCol, col+backCol):
			
			window = padded_image[(x - backRow):(x + 1 + backRow), (y - backCol):(y + 1 + backCol), 0]

			intesity1 = np.sum(filt * window, axis=(0, 1)) 
			intesity1 = intesity1/factor	
			
			intesity2 = np.sum(filt2 * window, axis=(0, 1)) 
			intesity2 = intesity2/factor	
			intesity = 	round(math.sqrt(intesity1**2+intesity2**2))
			if intesity > 255:
				intesity = 255
			if intesity < 0:
				intesity = 0
			image[x - backRow][y - backCol] = intesity
	cv2.imshow('image',image)
	cv2.imwrite(dest, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows() #optimazed using numpy







# Filters options (In a further implementation could used a file to import the filters)
filters = []
factors = []
filters.insert(0, [[ 0, 0,-1, 0, 0], 
	    [ 0,-1,-2,-1, 0],
   	    [-1,-2,16,-2,-1],
   	    [ 0,-1,-2,-1, 0],
   	    [ 0, 0,-1, 0, 0]])
factors.insert(0, 1)

filters.insert(1, [[ 1, 4, 6, 4, 1], 
	  [ 4, 16, 24, 16, 4],
   	  [ 6, 24, 36, 24, 6],
   	  [ 4, 16, 24, 16, 4],
    	  [ 1, 4, 6, 4, 1]])
factors.insert(1, 256)

filters.insert(2, [ [ -1, 0, 1],
			   	    [ -2, 0, 2],
			   	    [ -1, 0, 1]])
factors.insert(2, 1)

filters.insert(3, [ [ -1, -2, -1],
   	    		    [ 0, 0, 0,],
   	    			[ 1, 2, 1]])
factors.insert(3, 1)

filters.insert(4, [ [ -1, -1,-1],
   	  				[ -1, 8, -1],
   	    			[ -1, -1,-1]])
factors.insert(4, 1)

filters.insert(5, [ [ 1, 1, 1], 
	    			[ 1, 1, 1],
   	    			[ 1, 1, 1]])
factors.insert(5, 9)

filters.insert(6, [ [ -1, -1, 2], 
	   				[ -1, 2, -1],
   	    			[ 2, -1, -1]])
factors.insert(6, 1)

filters.insert(7, [ [ 2, -1, -1], 
	    			[ -1, 2, -1],
   	   				[ -1, -1, 2]])
factors.insert(7, 1)


filters.insert(8, [ [ 1, 0, 0, 0, 0, 0, 0, 0, 0],
   	    			[ 0, 1, 0, 0, 0, 0, 0, 0, 0],
   	    			[ 0, 0, 1, 0, 0, 0, 0, 0, 0],
   	    			[ 0, 0, 0, 1, 0, 0, 0, 0, 0],
   	    			[ 0, 0, 0, 0, 1, 0, 0, 0, 0],
   	    			[ 0, 0, 0, 0, 0, 1, 0, 0, 0],
   	    			[ 0, 0, 0, 0, 0, 0, 1, 0, 0],
   	    			[ 0, 0, 0, 0, 0, 0, 0, 1, 0],
   	    			[ 0, 0, 0, 0, 0, 0, 0, 0, 1]])
factors.insert(8, 9)

filters.insert(9, [ [ -1, -1,-1, -1, -1], 
				    [ -1, 2, 2, 2, -1],
			   	    [ -1, 2, 8, 2, -1],
			   	    [ -1, 2, 2, 2, -1],
			   	    [ -1, -1,-1, -1, -1]])
factors.insert(9, 8)

filters.insert(10, [[ -1, -1, 0], 
					[ -1, 0, 1],
			   	    [ 0, 1, 1]])
factors.insert(10, 1)
   		  
 
img2 = cv2.imread(args.image, 1)
if img2 is None:
	raise Exception("Invalid image")

#for i in range(11):
#	img2 = cv2.imread(args.image, 1)
#	convolve2(img2, filters[i], factors[i], ("h"+str(i+1)+args.destiny))

if args.filter > 0 and args.filter <= 11:
	convolve2(img2, filters[args.filter - 1], factors[args.filter - 1], args.destiny)
elif args.filter == 12:
	convolveSum(img2, filters[2], filters[3], 1, args.destiny)
else:
	raise Exception("Invalid filter")



