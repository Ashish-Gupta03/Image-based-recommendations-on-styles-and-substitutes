import json
import cv2
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
from sklearn.cluster import KMeans
# from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from scipy.spatial.distance import mahalanobis

# plt.figure(figsize = (11,10))
# gs1 = gridspec.GridSpec(11, 10)
# gs1.update(wspace=0, hspace=1)
sift = cv2.xfeatures2d.SIFT_create()
# 	g = gzip.open(path,'r')
# 	for l in g:
# 		yield eval(l)
# sift = cv2.xfeatures2d.SIFT_create()
# 'tshirts','jeans','sweat_shirts','shoes','googles','ties','watches','shirts','all'
trainPath = ['tshirts','jeans','sweat_shirts','shoes','googles','ties','watches','shirts']

path = os.path.join(os.getcwd(),'static')
path = os.path.join(path,'images')
path = os.path.join(path,'test')

for k in range(len(trainPath)):
	train_path = os.path.join(path,trainPath[k])

	print ('trainpath ',train_path)
	def load_images():
		count = 0
		imagesList = []
		for i in os.listdir(train_path):
			im1 = cv2.imread(os.path.join(train_path,i))
			imagesList.append(im1)
		return imagesList

	trainData = load_images()
	# print ('tri ',trainData)
	num_images = len(trainData)
	# ==================================   Bag of Visual Words  =================================================================
	desc_list = []
	for img in trainData:
		kp,des = sift.detectAndCompute(img,None)
		desc_list.append(des)

	desc_vStack = np.concatenate(desc_list,axis=0)

	n_components = 69
	pca = PCA(n_components=n_components)
	desc_vStack = pca.fit_transform(desc_vStack)
	# ================================	perform kmeans  =======================================
	n_clusters = 20
	kmeans = KMeans(n_clusters=n_clusters,random_state=0)
	desc_predict = kmeans.fit_predict(desc_vStack)
	# ==================================== BOVW end =============================

	# Generate histogram corresponding to frequency of each cluster(visual word)
	# like there are 50 pixels with 250 gray levels histogram is orderless(distribution)
	def gen_hist(n_clusters,num_images,desc_list,desc_predict):
		hist = np.array([np.zeros(n_clusters) for i in range(num_images)])
		init = 0
		for i in range(num_images):
			for j in range(len(desc_list[i])):
				idx = desc_predict[init+j]
				hist[i][idx] += 1
			init += 1

		return hist

	hist = gen_hist(n_clusters,num_images,desc_list,desc_predict)
	hist = hist.tolist()

	##############################################################################################
	distances = euclidean_distances(hist, hist)
	distances = distances.tolist()
	# print ('distances ',distances)

	#save distances
	joblib.dump(distances, 'distances'+str(k+1)+'.pkl')

	cluster_assignment = []
	for i in range(len(distances)):
		cluster_assignment.append(np.argsort(distances[i])[1:11])
	# print ('cluster_assignment ',cluster_assignment)

	#save cluster_assignment
	joblib.dump(cluster_assignment, 'cluster_assignment'+str(k+1)+'.pkl')

	# plt.imshow(cv2.cvtColor(trainData[5],cv2.COLOR_BGR2RGB))
	# plt.show()
	# k = 1
	# for i in cluster_assignment[5]:
	# 	plt.subplot(2,5,k)
	# 	plt.imshow(cv2.cvtColor(trainData[i], cv2.COLOR_BGR2RGB))
	# 	# plt.title(kmeans.labels_[i])
	# 	# plt.tight_layout()
	# 	plt.xticks([])
	# 	plt.yticks([])
	# 	k += 1
	# plt.show()	

###############################################################################################	
# cov = np.cov(hist, rowvar=False)
# nn = NearestNeighbors(algorithm='brute',n_neighbors=10, metric='mahalanobis',metric_params=dict(V=cov))
# distances,indices = nn.fit(hist).kneighbors(hist)

# cluster_assignment = []
# for i in range(len(distances)):
# 	cluster_assignment.append(np.argsort(distances[i])[1:11])
# # print ('cluster_assignment ',cluster_assignment)

# plt.imshow(cv2.cvtColor(trainData[50], cv2.COLOR_BGR2RGB))
# plt.show()
# k = 1
# for i in cluster_assignment[50]:
# 	plt.subplot(2,5,k)
# 	plt.imshow(cv2.cvtColor(trainData[i], cv2.COLOR_BGR2RGB))
# 	plt.xticks([])
# 	plt.yticks([])
# 	k += 1
# plt.show()
################################################################################################


# hist = np.asarray(hist)

# covar = np.cov(hist, rowvar=0)
# if(hist.shape[1:2]==(1,)):
#     invcovar = np.linalg.pinv(covar.reshape(1,1))
# else:
#     invcovar = np.linalg.pinv(covar)
	    
# dis = []
# finDis = []
# for i in hist:
# 	dis = []
# 	for j in hist:
# 		if np.array_equal(i,j) == False:
# 			dis.append(mahalanobis(i,j,invcovar))
# 	finDis.append(dis)

# joblib.dump(finDis, 'finDis'+str(k)+'.pkl')	
	# cov = np.cov(hist, rowvar=False)
	# nn = NearestNeighbors(algorithm='brute',n_neighbors=10, metric='mahalanobis',metric_params=dict(V=cov))
	# distances,indices = nn.fit(hist).kneighbors(hist)

# cluster_assignmentN = []
# for i in range(len(finDis)):	
# 	cluster_assignmentN.append(np.argsort(finDis[i])[1:11])

# joblib.dump(cluster_assignmentN, 'cluster_assignmentN'+str(k)+'.pkl')