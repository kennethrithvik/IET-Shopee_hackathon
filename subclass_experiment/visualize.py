from keras.models import Sequential,save_model,load_model

classifier = load_model('./output_models/cnn_867.h5')

from ann_visualizer.visualize import ann_viz
#import ann_visualizer
#Build your model here
ann_viz(classifier)