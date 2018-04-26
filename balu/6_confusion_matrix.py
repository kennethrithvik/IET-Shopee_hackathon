
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
model = load_model('../models/inception_best.h5')
print model.summary()

folders = glob('../data/Kfold/all/test/*')
print folders

X = []
y_actual = []
y_pred = []

for i,folder in enumerate(folders):
	print i
	actual_label = folder.split('/')[-1]
	images = glob(''.join([folder,'/*.jpg']))
	X += images
	y_actual += [i for img in images]

print len(y_actual)
print len(X)


def processBatch(batch):
	images = [cv2.imread(path) for path in batch]
	images = [cv2.resize(img,(img_width,img_height)) for img in images]
	images = np.array(images)
	images = images/255.0

	global model
	output = model.predict(images,batch_size=len(images))
	return np.argmax(output,axis=1)

batches = [X[i:i+batch_size] for i  in range(0, len(X), batch_size)]

for batch in batches:
	output = processBatch(batch)
	output = output.tolist()

	y_pred += output
	print len(y_pred)

df = pd.DataFrame()
df['y_actual'] = pd.Series(y_actual)
df['y_pred'] = pd.Series(y_pred)
df.to_csv('data/csv/results.csv',index=False)

df = pd.read_csv('data/csv/results.csv')
from sklearn.metrics import confusion_matrix,accuracy_score
print accuracy_score(df['y_actual'],df['y_pred'])
cm = confusion_matrix(df['y_actual'],df['y_pred'])
print cm

