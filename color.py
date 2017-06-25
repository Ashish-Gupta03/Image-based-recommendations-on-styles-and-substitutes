import numpy as np
import argparse
import cv2
import os 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

boundaries = [
	([50, 10, 10], [255, 20, 20]),
	([10, 50, 10], [20, 255, 20]),
	([10, 10, 50], [20, 20, 255])
	# ([25, 146, 190], [62, 174, 250]),
	# ([103, 86, 65], [145, 133, 128]),
	# ([110,50,50],[130,255,255])
]
blueList = []
greenList = []
redList = []
trainData = ['sweat_Shirt']


for i in trainData:
	image_file = os.listdir(i)
	for img in image_file:
			img_file = os.path.join(i,img)			
			im = cv2.imread(img_file)
			
			# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
			# im = im.reshape((im.shape[0] * im.shape[1], 3))
			# clt = KMeans(n_clusters = 1)
			# clt.fit(im)

			# hist = utils.centroid_histogram(clt)
			# bar = utils.plot_colors(hist, clt.cluster_centers_)
			 
			# # show our color bart
			# plt.figure()
			# plt.axis("off")
			# plt.imshow(bar)
			# plt.show()
			# im = np.reshape(im,(100,100,3))		
			j = 0
			for (lower, upper) in boundaries:
				count = 0

				lower = np.array(lower, dtype = "uint8")
				upper = np.array(upper, dtype = "uint8")

				mask = cv2.inRange(im, lower, upper)
				output = cv2.bitwise_and(im, im, mask = mask)
				# print ('output is ',output)
				for p in mask:
					for q in p:
						if q == 255:
							count += 1
				if count > 0 and j == 0:
					blueList.append((img_file,count))
				elif count > 0 and j == 1:
					greenList.append((img_file,count))
				elif count > 0 and j == 2:
					redList.append((img_file,count))
				j += 1
				# if mask[0][0] == 255:
				# 	cv2.imshow("images", np.hstack([im, output]))
				# 	cv2.waitKey(0)

blueList.sort(key=lambda x: x[1],reverse=True)
greenList.sort(key=lambda x: x[1],reverse=True)
redList.sort(key=lambda x: x[1],reverse=True)
print ('blueList ',blueList)
print ('greenList ',greenList)
print ('redList ',redList)
for (i,j) in blueList[:5]:
	im = cv2.imread(i)
	cv2.imshow('blue im is ',im)
	cv2.waitKey(0)

for (i,j) in greenList[:5]:
	im = cv2.imread(i)
	cv2.imshow('green im is ',im)
	cv2.waitKey(0)

for (i,j) in redList[:5]:
	im = cv2.imread(i)
	cv2.imshow('red im is ',im)
	cv2.waitKey(0)		