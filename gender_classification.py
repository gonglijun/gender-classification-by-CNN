import os
import numpy as np
import sys
import Image as image

caffe_root = './caffe-master/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('/home/gong/Downloads/caffe-master/python')
import caffe

mean_filename='./mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]


gender_net_pretrained='./gender_train_net.caffemodel'
gender_net_model_file='./deploy_gender.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=127,
                       image_dims=(128, 128))

gender_list=['Male','Female']

## ===============read the images in folder './image' ==============##
#path = './image'
#images = os.listdir(path)
#n = 1;
#for file in images:
#    if not os.path.isdir(file):
#        example_image = open(path + "/" + file)
#        input_image = caffe.io.load_image(example_image)
 #       _ = plt.imshow(input_image)
        
        #prediction = gender_net.predict([input_image]) 
        #myfile = open('./results/results.txt', 'a')
        #myfile.write(os.path.basename(file) + ' ')
        #myfile.write(gender_list[prediction[0].argmax()] + '\n')
        #n = n + 1
        #myfile.close()
        

example_image = './example_image.jpeg'
input_image = caffe.io.load_image(example_image)
#_ = plt.imshow(input_image)
        
prediction = gender_net.predict([input_image]) 
print 'predicted gender:', gender_list[prediction[0].argmax()]
        
