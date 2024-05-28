from __future__ import print_function
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2
# input image dimensions
img_rows, img_cols = 50, 50
input_dim = 50
batch_size = 50
num_classes = 3
epochs = 15

trainImagePath ='C:/Users/sruth/PycharmProjects/project2/project2/project/img_train/*.jpg'
testImagePath  = 'C:/Users/sruth/PycharmProjects/project2/project2/project/img_test/*.jpg'
trainAnnotationPath = 'C:/Users/sruth/PycharmProjects/project2/project2/project/annoTrain'
testAnnotaionPath = 'C:/Users/sruth/PycharmProjects/project2/project2/project/annoTest'
annotationPath = 'C:/Users/sruth/PycharmProjects/project2/project2/project/Annotations'

#reading bounding box
def xml_to_csv(path):
    classlist = ['ship', 'vehicle', 'airplane']
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
              if any(x in member.find('name').text for x in classlist):
                #print("root.find('filename').text :", root.find('filename').text)
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member.find('name').text,
                         int(member.find('bndbox')[0].text),
                         int(member.find('bndbox')[1].text),
                         int(member.find('bndbox')[2].text),
                         int(member.find('bndbox')[3].text),
                         )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

###get the dataframe and output it.
annotation_path = annotationPath
xml_dataframe = xml_to_csv(annotation_path)
xml_dataframe.to_csv('sample_labels.csv')
xml_dataframe = xml_dataframe.to_numpy()

print('xml_dataframe.shape[0]',xml_dataframe.shape)
xml_dataframe_train =[]
xml_dataframe_test  =[]
image_paths_train = glob.glob(trainImagePath)
for imagefile in image_paths_train:
    base = os.path.basename(imagefile)
    image_name = os.path.splitext(base)[0]
    for l in range(0, xml_dataframe.shape[0]):
        filename = os.path.splitext(xml_dataframe[l][0])[0]

        if image_name == filename:
            print('image_name :', image_name)
            print('filename :', filename)
            xml_dataframe_train.append(xml_dataframe[l])


xml_dataframe_train = np.array(xml_dataframe_train)

image_paths_test = glob.glob(testImagePath)
for imagetestfile in image_paths_test:
    testbase = os.path.basename(imagetestfile)
    test_image_name = os.path.splitext(testbase)[0]
    for l in range(0, xml_dataframe.shape[0]):
        filename_test = os.path.splitext(xml_dataframe[l][0])[0]
        if test_image_name == filename_test:
            xml_dataframe_test.append(xml_dataframe[l])
xml_dataframe_test = np.array(xml_dataframe_test)

print('xml_dataframe_train.shape[0]',xml_dataframe_train.shape)
print('xml_dataframe_test.shape[0]',xml_dataframe_test.shape[0])

y_train=[]
y_test =[]
for i in range(0,xml_dataframe_train.shape[0]):
    single_object = xml_dataframe_train[i]
    bb0_pred = single_object[4:8]
    classname=single_object[3]
    y_train.append(classname)
y_train= np.array(y_train)
print('y_train :',y_train.shape)
for i in range(0,xml_dataframe_test.shape[0]):
    single_object = xml_dataframe_test[i]
    bb0_pred = single_object[4:8]
    classname=single_object[3]
    y_test.append(classname)
y_test= np.array(y_test)
print('y_test :',y_test.shape)

# Reading training and test images
# TRAINING IMAGESC:
x_train = []  #trainingImages
image_paths_train = glob.glob(trainImagePath)
for imagefile in image_paths_train:
    base = os.path.basename(imagefile)
    image = Image.open(imagefile).resize((800, 800))
    image = np.asarray(image) / 255.0
    image_name = os.path.splitext(base)[0]
    for l in range(0, xml_dataframe_train.shape[0]):
        filename = os.path.splitext(xml_dataframe_train[l][0])[0]
        if image_name == filename:
            bb0_pred = xml_dataframe_train[l][4:8]
            img = image
            img = img[bb0_pred[1]:bb0_pred[3], bb0_pred[0]:bb0_pred[2]]
            img = cv2.resize(img, (input_dim, input_dim))
            x_train.append(img)
x_train = np.array(x_train)
print('x_train image read successful')
print('x_train :',x_train.shape)

# TEST IMAGES
x_test =[]
image_paths_test = glob.glob(testImagePath)
for imagetestfile in image_paths_test:
    test_base_name = os.path.basename(imagetestfile)
    testimage = Image.open(imagetestfile).resize((800, 800))
    testimage = np.asarray(testimage) / 255.0
    test_image_name = os.path.splitext(test_base_name)[0]
    for l in range(0, xml_dataframe_test.shape[0]):
        test_filename = os.path.splitext(xml_dataframe_test[l][0])[0]
        if test_image_name == test_filename:
            bb0_pred = xml_dataframe_test[l][4:8]
            testImg = testimage
            testImg = testImg[bb0_pred[1]:bb0_pred[3], bb0_pred[0]:bb0_pred[2]]
            testImg = cv2.resize(testImg, (input_dim, input_dim))
            x_test.append(testImg)
x_test = np.array(x_test)
print('x_test image read successful')
print('x_test :',x_test.shape)

#the channels last branch "else" is more common in image processing.
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
    print('input shape1 : ',input_shape)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols,3)
    print('input shape2 : ', input_shape)

# you can convert labels to numbers with dictionary
print('y_train :',y_train)
lables = ['ship', 'vehicle', 'airplane']
dict = { 'ship':0,  'vehicle':1,  'airplane':2}
train_labels = np.vectorize(dict.get)(y_train)
test_labels = np.vectorize(dict.get)(y_test)
print('y_train :',y_train)
print('train_labels: ',train_labels)
# convert class vectors to binary class matrices
print(train_labels)
print(num_classes)
y_train = keras.utils.to_categorical(train_labels, num_classes)
y_test = keras.utils.to_categorical(test_labels, num_classes)
print('y_train :',y_train)
print('num_classes :',num_classes)

#standard architecture for CNN, multiple convolutional layers, pooling layer to reduce dimensions
#of course flatten before using these features to the "MLP" network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(history.history.keys())

#Plot it - we can see "convergence"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])  #if validation_split>0
plt.title('Model TRAIN accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) #if validation_split>0
plt.title('Model TRAIN loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()