# nasnet-tensorflow

A nasnet in tensorflow based on tensorflow [slim](https://github.com/tensorflow/models/tree/master/research/slim) library.


## About Nasnet and this repository

Nasnet is so far the state-of-the-art image classification architecture on ImageNet dataset (ArXiv release date is 21 Jul. 2017), the single crop accuracy for nasnet-large model is reported to be 82.7. For details of nasnet, please refer to paper [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) by Barret Zoph etc.

With this repo., you should be able to:

- Train a nasnet with customized dataset for image classification task from scratch. (If you want)

- Finetune nasnet (nasnet-a-large, nasnet-a-mobile) from ImageNet pre-train model for image classification task.

- Test and evaluate the model you have trained.

- Deploy the model for your application or transfer the feature extractor to other tasks such as object detection. (By yourself)

Suitable for those who have solid CNN knowledge, python and tensorflow background. For those who have less background, [tensorflow slim walk through tutorial](https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb) should be a good start.


## Dependencies
tensorflow >= 1.4.0

tf.contrib.slim

numpy


## Usage
### Clone the repo and enter workspace.
```shell
git clone https://github.com/yeephycho/nasnet-tensorflow.git
cd nasnet-tensorflow
mkdir train pre-trained
```

### Download and converting to TFRecord format (This part is the same as tf.slim tutorial)
Many people would be interested in training Nasnet with their own data. I'm not sure whether it's a good idea to promote my repo. by using a dataset that provided by google's tutorial. Many people see the tfrecord generation code is a copy of tensorflow's solution, they just give up or send me an e-mail to ask how to train on customized dataset as I promised. However, if you spend some time on code, you would be able to find out that it may not be very easy to hard coding the tfrecord generation script by yourself but it's really easy for you to modify the template code and flower dataset is a very very good template for you to modify. So, before sending me e-mail, please spend half a hour on the following scripts:
```shell
train_image_classifier.py
download_and_convert_data.py
datasets/dataset_factory.py
datasets/download_and_convert_flowers.py
datasets/flowers.py
```
Just by modifing a few characters, you would be able to turn your own dataset into tfrecords.

The following instruction will lead you to generate tutorial tfrecords.

For each dataset, we'll need to download the raw data and convert it to
TensorFlow's native
[TFRecord](https://www.tensorflow.org/versions/r0.10/api_docs/python/python_io.html#tfrecords-format-details)
format. Each TFRecord contains a
[TF-Example](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/core/example/example.proto)
protocol buffer. Below we demonstrate how to do this for the Flowers dataset.

```shell
$ DATA_DIR=/tmp/data/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```

When the script finishes you will find several TFRecord files created:

```shell
$ ls ${DATA_DIR}
flowers_train-00000-of-00005.tfrecord
...
flowers_train-00004-of-00005.tfrecord
flowers_validation-00000-of-00005.tfrecord
...
flowers_validation-00004-of-00005.tfrecord
labels.txt
```

These represent the training and validation data, sharded over 5 files each.
You will also find the `$DATA_DIR/labels.txt` file which contains the mapping
from integer labels to class names.

I provide a user friendly version of tfrecord generation solution here.
All you need to modify is located at 
```shell
datasets/customized.py
```
Line 36, the number of training set and validation (test) set.

Line 39, the number of total classes.

```shell
datasets/convert_customized.py
```
Line 61, the number of validation (test) set.


```shell
# Create directories that name after the labels, then put the images under the label folders.
# ls /path/to/your/dataset/
# label0, label1, label2, ...
# ls /path/to/your/dataset/label0
# label0_image0.jpg, label0_image1.jpg, ...
#
# Image file name doesn't really matter.
DATASET_DIR=/path/to/your/own/dataset/

# Convert the customized data into tfrecords. Be noted that the dataset_name must be "customized"!
python convert_customized_data.py \
    --dataset_name=customized \
    --dataset_dir="${DATASET_DIR}"
```

### Train from scratch
```shell
DATASET_DIR=/tmp/data/flowers # /path/to/your/own/dataset/
TRAIN_DIR=./train

# For Nasnet-a-mobile
# --dataset_name=customized
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=nasnet_mobile

# For Nasnet-a-large
# --dataset_name=customized
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=nasnet_large
```

### Finetune from ImageNet pre-trained checkpoint
```shell
# This script will download pre-trained model from google, mv the file to pre-trained folder and unzip the file.
sh download_pretrained_model.sh

DATASET_DIR=/tmp/data/flowers # /path/to/your/own/dataset/
TRAIN_DIR=./train

# For Nasnet-a-mobile
# --dataset_name=customized
CHECKPOINT_PATH=./pre-trained/nasnet-a_mobile_04_10_2017/model.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=nasnet_mobile \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=final_layer,aux_7 \
    --trainable_scopes=final_layer,aux_7

# For Nasnet-a-large
# --dataset_name=customized
CHECKPOINT_PATH=./pre-trained/nasnet-a_large_04_10_2017/model.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=nasnet_large \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=final_layer,aux_11 \
    --trainable_scopes=final_layer,aux_11
```

### Evaluation
A nasnet finetuned model for flowers dataset can be downloaded [here](https://drive.google.com/open?id=1l_hhQoE6T4rc69OpRMJ8geQXzgTnXqUC) from google drive.

```shell
# Please specify the model.ckpt-xxxx file by yourself, for example
CHECKPOINT_FILE=./train/model.ckpt-29735

# For Nasnet-a-mobile
# --dataset_name=customized
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=/tmp/data/flowers \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --model_name=nasnet_mobile

# For Nasnet-a-large
# --dataset_name=customized
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=/tmp/data/flowers \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --model_name=nasnet_large
```

### Visualize the training progress
```shell
tensorboard --logdir=./train
```


### Inference

#### Export static computation graph

Find file export_inference_graph.py

Change line 112, num_classes=5 (For flowers dataset, change this parameter according to your own dataset)


```shell
# For large model
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=nasnet_large \
  --output_file=./inference/nasnet_large_inf_graph.pb

# For mobile model
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=nasnet_mobile \
  --output_file=./inference/nasnet_mobile_inf_graph.pb
```

#### Freeze the graph and checkpoint model

```shell
# For large model
python freeze_graph.py \
  --input_graph=./inference/nasnet_large_inf_graph.pb \
  --input_checkpoint=./train/model.ckpt-16547 \
  --input_binary=true \
  --output_graph=./inference/frozen_nasnet_large.pb \
  --output_node_names=final_layer/predictions

# For mobile model
python freeze_graph.py \
  --input_graph=./inference/nasnet_mobile_inf_graph.pb \
  --input_checkpoint=./train/yourOwnModel.ckpt-xxxx \
  --input_binary=true \
  --output_graph=./inference/frozen_nasnet_mobile.pb \
  --output_node_names=final_layer/predictions
```

#### Inference demo
##### Note: 
1. Make sure input image has a size that larger than 331 by 331, no resize operation for current input, you can add if you want.

2. Only support jpeg for inference demo.

3. Five crops inference strategy is adopted.

```shell
python image_inference.py
```

#### Output
![Inference demo output](https://github.com/yeephycho/nasnet-tensorflow/blob/master/demo/output/demo_output.png?raw=true "Show result")



## Reference
[Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

[tf.contrib.slim](https://github.com/tensorflow/models/tree/master/research/slim)

