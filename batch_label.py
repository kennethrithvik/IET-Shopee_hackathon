import numpy as np
import os
import pandas as pd
from glob import glob
import tensorflow as tf

batch_size = 128
img_width, img_height = 224,224

test_images = glob('./data/sample_test/*.jpg')
# Unpersists graph from file
with tf.gfile.FastGFile("./models/nasnet_mobile_inf_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def processBatch(batch):
    output=[]
    for path in batch:
        image_data = tf.gfile.FastGFile(path, 'rb').read()
        #images = [cv2.imread(path) for path in batch]
        #images = [cv2.resize(img,(img_width,img_height)) for img in images]
        #images = np.array(images)
        #images = images/255.0
        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_layer/predictions:0')   
            predictions = sess.run(softmax_tensor,{'input:0': image_data})      
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            output.append(top_k[0])
    ids = [path.split('.')[0] for path in batch]
    ids = [path.split('_')[-1] for path in ids]
    ids = [int(path) for path in ids]

	#global model
	#output = model.predict(images,batch_size=len(images))
	
    return ids,output

batches = [test_images[i:i+batch_size] for i  in range(0, len(test_images), batch_size)]

X = []
y = []

for batch in batches:
	ids,output = processBatch(batch)
	ids = list(ids)
	print (ids,output)
	X += ids
	y += output

df = pd.DataFrame()
df['id'] = pd.Series(X)
df['category'] = pd.Series(y)
df = df.sort_values(by='id')
#print(df)
df.to_csv('output/evaluateTest.csv',index=False)

