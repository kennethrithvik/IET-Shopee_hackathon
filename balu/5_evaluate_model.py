
import numpy as np
import cv2
from glob import glob
import pandas as pd

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from keras.models import load_model

batch_size = 128
epochs = 5
img_width, img_height = 139,139
evaluate_folder = '../data/test'
n_classes = 18

test_images = glob('../data/test/*.jpg')
model = load_model('../models/NASNet-large.h5')

def processBatch(batch):
	images = [cv2.imread(path) for path in batch]
	images = [cv2.resize(img,(img_width,img_height)) for img in images]
	images = np.array(images)
	images = images/255.0

	ids = [path.split('.')[0] for path in batch]
	ids = [path.split('_')[-1] for path in ids]
	ids = [int(x) for x in ids]

	global model
	output = model.predict(images,batch_size=len(images))
	
	return ids,np.argmax(output,axis=1)

batches = [test_images[i:i+batch_size] for i  in range(0, len(test_images), batch_size)]


X = []
y = []

for batch in batches:
	ids,output = processBatch(batch)
	ids = list(ids)
	output = output.tolist()
	# print output
	X += ids
	y += output

	print(len(y))

df = pd.DataFrame()
df['id'] = pd.Series(X)
df['category'] = pd.Series(y)
df = df.sort_values('id')
df.to_csv('evaluate.csv',index=False)