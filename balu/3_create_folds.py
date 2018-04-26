import pandas as pd
import cv2
from glob import glob
import os
import shutil

def createFolder(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)


folders = glob('../data/train/*')
for folder in folders:
	name = folder.split('/')[-1]
	print (name)
	for i in range(0,4):

		df = pd.read_csv('../data/csv/train_fold_'+str(i)+'.csv')
		paths = df[df['label']==name]['path']
		
		images = [cv2.imread(x) for x in paths]
		images_names = [path.split('/')[-1] for path in paths]

		createFolder('../data/Kfold/fold'+str(i)+'/train/'+name)
		for j,img in enumerate(images):
			write_to = '../data/Kfold/fold'+str(i)+'/train/'+name+'/'+images_names[j]
			print(write_to)
			cv2.imwrite(write_to,img)


		df = pd.read_csv('../data/csv/test_fold_'+str(i)+'.csv')
		paths = df[df['label']==name]['path']
		
		images = [cv2.imread(x) for x in paths]
		images_names = [path.split('/')[-1] for path in paths]

		createFolder('../data/Kfold/fold'+str(i)+'/test/'+name)
		for j,img in enumerate(images):
			write_to = '../data/Kfold/fold'+str(i)+'/test/'+name+'/'+images_names[j]
			print (write_to)
			cv2.imwrite(write_to,img)

