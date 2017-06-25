import numpy as np
import pandas as pd
from flask import Flask , abort , jsonify,request
import json
from flask_cors import CORS,cross_origin
from sklearn.externals import joblib
import numpy as np
import os
from os.path import join
from flask import render_template
import cv2
import random

train_path = ['home','tshirts','jeans','sweat_shirts','shoes','googles','ties','watches','shirts']
num_files=[]

path = join(os.getcwd(),'static')
path=join(path,'images')
path=join(path,'test')

#for images path to be displayed
images=[]
for j in range(len(train_path)):
	sent_path="..\static\images\\test\\"+train_path[j]
	dir_path=join(path,train_path[j])
	sent_image=[]
	num_files.append(len(os.listdir(dir_path)))
	for i in os.listdir(dir_path):
		sent_image.append(join(sent_path,i))
	images.append(sent_image)


app=Flask(__name__)
CORS(app)

@app.route("/",methods=['GET','POST'])
def hello():
	return render_template('index.html')

@app.route("/cat",methods=['GET','POST'])
def cat():
	directory=int(request.args['id'])
	print ('directory ',directory)
	nameofdir=train_path[directory]
	print ('nameofdir ',nameofdir)
	img_path=[]
	seq=[]
	for i in range(12):
		num=int(random.randrange(1,num_files[directory]))
		seq.append(num)
		img_path.append(images[directory][num])
	return render_template('cat.html',data=img_path,seq=seq,name=nameofdir,did=directory)

@app.route("/contact",methods=['GET','POST'])
def contact():
	return render_template('contact.html')

@app.route("/api",methods=['GET','POST'])
def make_predict():
	print('start substitute')
	imgno=int(request.args['id'])

	print ('imgno is ',imgno)
	dirid=int(request.args['dirid'])
	print ('dirid ',dirid)
	nameofdir=train_path[dirid]
	print ('nameofdir ',nameofdir)
	li = []
	temp = []
	cluster_assignment = []
	did = []
	###################################
	#loading model
	# distances=joblib.load('distances'+str(dirid)+'.pkl')
	cluster_assignment=joblib.load('cluster_assignment'+str(dirid)+'.pkl')
	cluster_assignmentN=joblib.load('cluster_assignmentN.pkl')
	###################################
	#which category
	res=images[dirid][imgno]
	li.append(res)
	temp.append(imgno)
	for i in cluster_assignment[imgno][:6]:
		did.append(dirid)
		li.append(images[dirid][i])
		temp.append(i)

	pathN = "..\static\images\\test\\all\\"

	for i in cluster_assignmentN[imgno][:6]:
		flag = 0
		for j in range(len(train_path)-1):
			if j != 'home':
				# print ('j is ',train_path[j+1])
				img_file = os.listdir(os.path.join(path,train_path[j+1]))
				for img in img_file:
					if img == str(i)+'.jpg':
						did.append(j+1)
						flag = 1
						break
			if flag == 1:
				break		
		li.append(pathN+str(i)+'.jpg')
		temp.append(i)		
	print(li)
	return render_template('details.html', list = li, seq = temp,did=did,name=nameofdir)

@app.route("/api2",methods=['GET','POST'])
def make_complimentary():
	print('start make complimentary')
	imgno=int(request.args['id'])
	print ('imgno is ',imgno)
	dirid=int(request.args['dirid'])
	print ('dirid ',dirid)
	nameofdir=train_path[dirid]
	print ('nameofdir ',nameofdir)

	li = []
	temp = []
	cluster_assignment = []
	did = []
	###################################
	#loading model
	cluster_assignment=joblib.load('cluster_assignment'+str(dirid)+'.pkl')
	cluster_assignmentN=joblib.load('cluster_assignmentN.pkl')
	###################################
	#which category
	k = 0
	for j,i in enumerate(os.listdir(os.path.join(path,train_path[dirid]))):
		if i == str(imgno)+'.jpg':
			k = j
			break
	# print ('k is ',k)
	print ('cluster is ',cluster_assignment[k])
	res=images[dirid][k]
	li.append(res)
	temp.append(k)
	for i in cluster_assignment[k][:6]:
		did.append(dirid)
		li.append(images[dirid][i])
		temp.append(i)

	pathN = "..\static\images\\test\\all\\"

	for i in cluster_assignmentN[imgno][:6]:
		flag = 0
		for j in range(len(train_path)-1):
			if j != 'home':
				# print ('j is ',train_path[j+1])
				img_file = os.listdir(os.path.join(path,train_path[j+1]))
				for img in img_file:
					if img == str(i)+'.jpg':
						did.append(j+1)
						flag = 1
						break
			if flag == 1:
				break		
		li.append(pathN+str(i)+'.jpg')
		temp.append(i)		
	print(li)
	return render_template('details.html', list = li, seq = temp,did=did,name=nameofdir)

@app.route("/contactPost",methods=['GET','POST'])
def contactPost():
	return render_template('contactPost.html')


if __name__ == '__main__':
	app.run(port=9005,debug=True)
