import tensorflow as tf
from google.protobuf.json_format import MessageToJson
i=0
for string_record in tf.python_io.tf_record_iterator("./tmp/train/customized_validation_00000-of-00005.tfrecord"):
    i+=1
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
    height = int(example.features.feature['image/height']
                                 .int64_list
                                 .value[0])
    
    width = int(example.features.feature['image/width']
                                .int64_list
                                .value[0])
    
    img_string = (example.features.feature['image/encoded']
                                  .bytes_list
                                  .value[0])
    
    annotation_string = (example.features.feature['image/class/label']
                                .int64_list
                                .value)
    
    print(annotation_string)
print(i)
