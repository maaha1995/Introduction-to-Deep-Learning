#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:29:31 2017

@author: mahalakshmimaddu
"""

import numpy as np
import os
from glob import glob 
from tqdm import tqdm
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split

def get_image(image_path, w=150, h=150):
    im = imread(image_path).astype(np.float)
    orig_h, orig_w = im.shape[:2]
    new_h = int(orig_h * w / orig_w)
    im = imresize(im, (new_h, w))
    margin = int(round((new_h - h)/2))
    return im[margin:margin+h]



#### Importing labels
    
print("Name: Charanya Sudharsanan\nPerson Number: 50245956\n")
print("Name: Mahalakshmi Maddu\nPerson Number: 50246769\n")
print("Name: Nitish Shokeen\nPerson Number: 50247681\n")
    
print("Importing Labels\n")
data_labels = np.genfromtxt("./list_attr_celeba.txt",dtype=None, skip_header = 2, usecols = 15)

#print("Importing Labels")
for i in range(len(data_labels)):
    if data_labels[i] < 0:
        data_labels[i] = 0

#### Converting images into array

data_images = glob(os.path.join("./img_align_celeba", "*.jpg"))
data_images = np.sort(data_images)
w, h = 150, 150  
data = np.zeros((len(data_images), 3, 150, 150), dtype = np.uint8)


print("creating training validation and test data from the images:\n")
for n, file_name in tqdm(enumerate(data_images)):
    images = get_image(file_name)
    y = np.array(images).reshape((3,150,150))
    data[n]=y


#### Splitting the data into training, test and validation

print("Dividing the data into training validation and test:\n");
x_training, x_test, y_training, y_test = train_test_split(data, data_labels, test_size=0.25)

x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5)

y_training = np_utils.to_categorical(y_training)
y_validation = np_utils.to_categorical(y_validation)
y_test = np_utils.to_categorical(y_test)

numberOfClasses = y_validation.shape[1]

seed = 9
np.random.seed(seed)


########### CNN 

print("creating the model:\n");

cnn = Sequential()
#cnn.add(Conv2D(32, (5, 5), input_shape=(3, 160, 160), activation='relu'))
cnn.add(Conv2D(32, (2, 2), input_shape=(3, 150, 150), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(16, (1, 1), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(1, 1)))
cnn.add(Dropout(0.4))
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dense(500, activation='relu'))
cnn.add(Dense(numberOfClasses, activation='softmax'))
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#### Model fitting and evaluating

print("fitting the model:\n")

cnn.fit(x_training, y_training, validation_data=(x_validation, y_validation), epochs=2, batch_size=20, verbose=2)

print("calculating the score:\n")

result = cnn.evaluate(x_test, y_test, verbose=0)

print("Accuaracy for Deep CNN Model:%",result[1]*100)

print("Error for Deep CNN : %.2f%%" % (100-result[1]*100))