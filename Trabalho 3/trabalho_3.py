import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import argparse
import math 
from argparse import RawTextHelpFormatter
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

parser.add_argument("-m", "--method", help="Thresholding method number: \n"
											"1 - to global thresholding (Threshold default = 128)\n"
											"2 - to Bernsen (Default: n = 50)\n"
											"3 - to Niblack (Default: n = 50, k = -0.05)\n"
											"4 - to Sauvola (Default: n = 50, k = 0.5, r = 125)\n"
											"5 - to Phansalskar (Default: n = 50, k = 0.25, r = 0.5, p = 2, q =10)\n"
											"6 - to Contrast (Default: n = 50)\n"
											"7 - to Mean (Default: n = 50)\n"
											"8 - to Median (Default: n = 50)", type=int, required=True)
parser.add_argument("-i", "--image", help="Image to apply the Thresholding", required=True)
parser.add_argument("-t", "--threshold", help="threshold to global thresholding", type=int)
parser.add_argument("-n", "--n", help="Windows size nxn to local thresholding", type=int)
parser.add_argument("-k", "--k", help="k to local thresholding", type=float)
parser.add_argument("-r", "--r", help="r to local thresholding", type=float)
parser.add_argument("-p", "--p", help="p to local thresholding", type=float)
parser.add_argument("-q", "--q", help="q to local thresholding", type=float)
args = parser.parse_args()

N = 50
THREHOLD = 128
K_NIBLACK = -0.05

K_SAUVOLA = 0.5
R_SAUVOLA = 125

K_PHANSALSKAR = 0.25
R_PHANSALSKAR = 0.5
P_PHANSALSKAR = 2
Q_PHANSALSKAR = 10

def globalThresholding(image, threshold):
	binary_min = image > threshold

	fig, ax = plt.subplots(2, 2, figsize=(10, 10))

	ax[0, 0].imshow(image, cmap=plt.cm.gray)
	ax[0, 0].set_title('Original')

	ax[0, 1].hist(image.ravel(), bins=256)
	ax[0, 1].set_title('Histogram')

	ax[1, 0].imshow(binary_min, cmap=plt.cm.gray)
	ax[1, 0].set_title('Thresholded (min)')

	ax[1, 1].hist(image.ravel(), bins=256)
	ax[1, 1].axvline(threshold, color='r')

	for a in ax[:, 0]:
	    a.axis('off')
	plt.show()

def localThresholding(image, threshold, n, m):
	image2 = image.copy()
	image = image.astype(np.int64)
	row = image.shape[0]
	col = image.shape[1]
	dimension = n*m
	backRow = math.floor((n-1)/2)
	backCol = math.floor((m-1)/2)
	padded_image = np.pad(image, [
	        (backRow, backRow),
	        (backCol, backCol)
	    ], "edge")
	pintou = True
	for x in range(row):
		for y in range(col):
			image[x][y]= threshold(padded_image[(x) : (x + 2*backRow + 1) , (y) : (y + 2*backCol + 1)], image[x][y])

	fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
	ax = axes.ravel()
	ax[0] = plt.subplot(1, 3, 1)
	ax[1] = plt.subplot(1, 3, 2)
	ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

	ax[0].imshow(image2, cmap=plt.cm.gray)
	ax[0].set_title('Original')
	ax[0].axis('off')

	ax[1].hist(image2.ravel(), bins=256)
	ax[1].set_title('Histogram')

	ax[2].imshow(image, cmap=plt.cm.gray)
	ax[2].set_title('Thresholded')
	ax[2].axis('off')

	plt.show()


def bersen (matrix, pixel):
	threshold = (np.max(matrix) + np.min(matrix))/2
	return np.where(pixel < threshold, 0, 255)

def niblack (matrix, pixel):
	threshold = np.mean(matrix) + np.std(matrix)*K_NIBLACK
	return np.where(pixel < threshold, 0, 255)

def sauvola (matrix, pixel):
	threshold = np.mean(matrix) *(1 + K_SAUVOLA *((np.std(matrix) / R_SAUVOLA) -1))
	return np.where(pixel < threshold, 0, 255)

def phansalskar (matrix, pixel):
	mean = np.mean(matrix)
	threshold = mean * (1 + (P_PHANSALSKAR * np.exp(-Q_PHANSALSKAR * mean)) + K_PHANSALSKAR *((np.std(matrix) / R_PHANSALSKAR) -1))
	return np.where(pixel < threshold, 0, 255)

def contrast (matrix, pixel):
	return np.where(((np.max(matrix) - pixel) > (pixel - np.min(matrix))), 0, 255)

def mean (matrix, pixel):
	return np.where(pixel < np.mean(matrix), 0, 255)

def median (matrix, pixel):
	return np.where(pixel < np.median(matrix), 0, 255)


image = cv2.imread(args.image, 0)
if image is None:
	raise Exception("Invalid image")

if args.threshold != None:
	THREHOLD = args.threshold

if args.n != None:
	N = args.n

if args.k != None:
	K_NIBLACK = args.k
	K_SAUVOLA = args.k
	K_PHANSALSKAR = args.k

if args.r != None:
	R_PHANSALSKAR = args.r
	R_SAUVOLA = args.r

if args.p != None:
	P_PHANSALSKAR = args.p

if args.q != None:
	Q_PHANSALSKAR = args.q



if args.method == 1:
	globalThresholding(image, THREHOLD)
elif args.method == 2:
	localThresholding(image, bersen, N, N)
elif args.method == 3:
	localThresholding(image, niblack, N, N)
elif args.method == 4:
	localThresholding(image, sauvola, N, N)
elif args.method == 5:
	localThresholding(image, phansalskar, N, N)
elif args.method == 6:
	localThresholding(image, contrast, N, N)
elif args.method == 7:
	localThresholding(image, mean, N, N)
elif args.method == 8:
	localThresholding(image, median, N, N)
else:
	raise Exception("Invalid method")

