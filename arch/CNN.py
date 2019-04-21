# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:25:00 2019

@author: jltsa
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import cv2
import random
import os
from keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        #Create instance of model
        model = Sequential()
        inputShape = (height, width, depth)
        #Add layers
        
        #First convolutional layer w/ 20 convolutional filters which are 5x5 pixels
        #Padding set to same so input size into the convolution is the same as output
        #Activation can be set in convolutiuonal layer or added separately
        model.add(Conv2D(filters=20, kernel_size=5, padding='same',
                         input_shape=inputShape, activation='relu'))
        #model.add(Activation('relu'))
        #Calculate the max values of pixels in a window size of 2x2 pixels
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #2nd covolutional layer
        model.add(Conv2D(filters=50, kernel_size=5, padding='same',
                        activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #Flatten layer serves as a connecter between convolution and densely connected layers
        #Flattens into a single vector
        model.add(Flatten())
        #Dense layer contains 500 fully connected nodes
        model.add(Dense(500, activation='relu'))
        #Can add activation layer separately
        #model.add(Activation('relu'))
        #Output will be number of classes, softmax will yield probability of each class
        model.add(Dense(classes, activation='softmax'))
        
        return model
    
class AlexNet:
    #optimizer to use should be 'sgd'
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        #input shape is 227x227x3
        inputShape = (height, width, depth)
        model.add(Conv2D(filters=96, kernel_size=11, strides=4,
                         input_shape=inputShape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=5, strides=1,
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Conv2D(filters=384, kernel_size=3, strides=1,
                         padding='same', activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=3, strides=1,
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        
        return model
    
def pre_process(width, height, path):
    """
    Resize and rescale images stored in image folder.
    
    Return pre-processed data and labels for the classes based
    on sub-directories in the image folder
    """
    #containers for pre-processed image data and class labels
    data = []
    labels = []

    #images directory contains 3 sub-directories: 'poison_ivy', 'poison_oak', 'poison_sumac'
    #randomly get image paths and shuffle them
    # current path 'C:\\Users\\jltsa\\Desktop\\Project_2\\images'
    image_paths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(image_paths)

    #preprocess images to width x height pixels as required for LeNet
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (width, height))
        image = img_to_array(image)
        data.append(image)
    
        #Extract class labels
        label = image_path.split(os.path.sep)[-2]
        if label == 'poison_ivy':
            label = 0
        elif label == 'poison_oak':
            label = 1
        else:
            label = 2
        labels.append(label)
       
    #Scal pixel intensities from 0 to 1
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)
    
    return data, labels