import cv2
import numpy as np
import sys
import argparse
import math 
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mask", help="mask number (1 to Floyd e Steinberg, 2 to Stevenson e Arce, 3 to Burkes, 4 to Sierra, 5 to Stucki, 6 to Jarvis, Judice e Ninke)", type=int, required=True)
parser.add_argument("-i", "--image", help="Image to apply the halftoning", required=True)
parser.add_argument("-d", "--destiny", help="Name of destiny file", required=True)
parser.add_argument("-g", "--gray", help="1 to gray", type=int)
parser.add_argument("-z", "--zigzag", help="1 to zigzag", type=int)
args = parser.parse_args()

args = parser.parse_args()

def halfGray (image, filt, name):
	img = image.copy()
	backRow = (len(filt))
	backCol = math.floor((len(filt[0])-1)/2)
	padded_image = np.pad(img, [
        (backRow, backRow),
        (backCol, backCol)
    ])
	row = img.shape[0]
	col = img.shape[1]
	img = img.astype(np.int64)
	padded_image = padded_image.astype(np.int64)
	filt = np.array(filt)
	for x in range(row):
		for y in range(col):
			img[x][y] = np.where(padded_image[x + backRow][y + backCol] < 128, 0, 255)
			erro = padded_image[x + backRow][y + backCol] - img[x][y]
			window = padded_image[(x + backRow):(x + backRow + backRow), (y):(y + 1 + backCol + backCol)]
			data = filt*erro
			padded_image[(x + backRow):(x + backRow + backRow), (y):(y + 1 + backCol + backCol)] += data.astype(np.int64)

			
				
						
	cv2.imshow('image',img.astype(np.uint8))
	cv2.imwrite(name, img.astype(np.uint8))
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def halfGrayZigzag (image, filt, name):
	img = image.copy()
	backRow = (len(filt))
	backCol = math.floor((len(filt[0])-1)/2)
	padded_image = np.pad(img, [
        (backRow, backRow),
        (backCol, backCol)
    ])
	row = img.shape[0]
	col = img.shape[1]
	img = img.astype(np.int64)
	padded_image = padded_image.astype(np.int64)
	filt = np.array(filt)
	for x in range(row):
		r = reversed(range(col))
		if (x % 2) == 0 :
			r = range(col)

		for y in r:
			img[x][y] = np.where(padded_image[x + backRow][y + backCol] < 128, 0, 255)
			erro = padded_image[x + backRow][y + backCol] - img[x][y]
			window = padded_image[(x + backRow):(x + backRow + backRow), (y):(y + 1 + backCol + backCol)]
			data = filt*erro
			padded_image[(x + backRow):(x + backRow + backRow), (y):(y + 1 + backCol + backCol)] += data.astype(np.int64)

			
				
						
	cv2.imshow('image',img.astype(np.uint8))
	cv2.imwrite(name, img.astype(np.uint8))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def half (image, filt, name):
	img = image.copy()
	backRow = (len(filt))
	backCol = math.floor((len(filt[0])-1)/2)
	padded_image = np.pad(img, [
        (backRow, backRow),
        (backCol, backCol),
        (0, 0)
    ])
	row = img.shape[0]
	col = img.shape[1]
	img = img.astype(np.int64)
	padded_image = padded_image.astype(np.int64)
	filt = np.array(filt)
	for x in range(row):
		for y in range(col):
			img[x][y] = np.where(padded_image[x + backRow][y + backCol] < 128, 0, 255)
			erro = padded_image[x + backRow][y + backCol] - img[x][y]
			window = padded_image[(x + backRow):(x + backRow + backRow), (y):(y + 1 + backCol + backCol)]	
			data = (np.stack((filt,)*3, axis=-1)*erro) 	
			padded_image[(x + backRow):(x + backRow + backRow), (y):(y + 1 + backCol + backCol)] += data.astype(np.int64)


			
				
						
	cv2.imshow('image',img.astype(np.uint8))
	cv2.imwrite(name, img.astype(np.uint8))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def halfZigzag (image, filt, name):
	img = image.copy()
	backRow = (len(filt))
	backCol = math.floor((len(filt[0])-1)/2)
	padded_image = np.pad(img, [
        (backRow, backRow),
        (backCol, backCol),
        (0, 0)
    ])
	row = img.shape[0]
	col = img.shape[1]
	img = img.astype(np.int64)
	padded_image = padded_image.astype(np.int64)
	filt = np.array(filt)
	for x in range(row):
		r = reversed(range(col))
		if (x % 2) == 0 :
			r = range(col)

		for y in r:
			img[x][y] = np.where(padded_image[x + backRow][y + backCol] < 128, 0, 255)
			erro = padded_image[x + backRow][y + backCol] - img[x][y]
			window = padded_image[(x + backRow):(x + backRow + backRow), (y):(y + 1 + backCol + backCol)]
			data = (np.stack((filt,)*3, axis=-1)*erro) 
			padded_image[(x + backRow):(x + backRow + backRow), (y):(y + 1 + backCol + backCol)] += data.astype(np.int64)

			
				
						
	cv2.imshow('image',img.astype(np.uint8))
	cv2.imwrite(name, img.astype(np.uint8))
	cv2.waitKey(0)
	cv2.destroyAllWindows()



# Filters options (In a further implementation could used a file to import the filters)
filters = []

filters.insert(0, [[ 0, 0, 7/16], 
					[ 3/16, 5/16, 1/16]])
filters.insert(1, [[ 0, 0, 0, 0, 0, 32/200, 0],
				   [ 12/200, 0, 26/200, 0, 30/200, 0, 16/200], 
				   [ 0, 12/200, 0, 26/200, 0, 12/200, 0],
				   [ 5/200, 0, 12/200, 0, 12/200, 0, 5/200]])

filters.insert(2, [[ 0, 0, 0, 8/32, 4/32], 
				   [ 2/32, 4/32, 8/32, 4/32, 2/32]])

filters.insert(3, [	[ 0, 0, 0, 5/32, 3/32],
					[ 2/32, 4/32, 5/32, 4/32, 2/32],
					[ 0, 2/32, 3/32, 2/32, 0]])

filters.insert(4, [	[ 0, 0, 0, 8/42, 4/42],
					[ 2/42, 4/42, 8/42, 4/42, 2/42],
					[ 1/42, 2/42, 4/42, 2/42, 1/42]])

filters.insert(5, [	[ 0, 0, 0, 7/48, 5/48],
					[ 3/48, 5/48, 7/48, 5/48, 3/48],
					[ 1/48, 3/48, 5/48, 3/48, 1/48]])
 
img2 = cv2.imread(args.image, 1)
if img2 is None:
	raise Exception("Invalid image")
#cv2.imshow('image',img2)
img = cv2.imread(args.image, 0)

if args.mask > 6:
	raise Exception("Invalid mask")

if args.gray == 1:
	if args.zigzag == 1:
		halfGrayZigzag(img, filters[args.mask-1], args.destiny)
	else:
		halfGray(img, filters[args.mask-1], args.destiny)
else:
	if args.zigzag == 1:
		halfZigzag(img2, filters[args.mask-1], args.destiny)
	else:
		half(img2, filters[args.mask-1], args.destiny)



