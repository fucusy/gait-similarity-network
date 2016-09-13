# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 09:42:56 2016

@author: liuzheng

    File for contributers to define their DIY models.
    It is appreciated if contributers follow the same parameter list.
"""
import sys
sys.path.append('../')

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta

from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization

import numpy as np


def inference(input_shape, classNum, weights_file=''):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(classNum))
    model.add(Activation('softmax'))
    
    if weights_file:
        model.load_weights(weights_file)
    
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def inference_xavier_prelu_sgd_224(input_shape, classNum, weights_file=''):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape, init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(32, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='valid', init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(256, 3, 3, border_mode='valid', init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(256, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='valid', init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(classNum))
    model.add(Activation('softmax'))
    
    if weights_file:
        model.load_weights(weights_file)
#    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model






















