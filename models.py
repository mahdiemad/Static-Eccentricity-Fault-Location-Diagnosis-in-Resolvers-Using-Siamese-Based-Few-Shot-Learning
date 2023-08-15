# -*- coding: utf-8 -*-
from keras.layers import Input, Conv2D,Conv1D, Lambda, merge, Dense, Flatten,MaxPooling2D,MaxPooling1D,Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import time
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_siamese_net(input_shape = (75,75,3)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # WDCNN
    convnet = Sequential()

    convnet.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='valid',input_shape=input_shape))
    convnet.add(MaxPooling2D((2,2)))

    convnet.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='valid'))
    convnet.add(MaxPooling2D((2,2)))

    convnet.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='valid'))
    convnet.add(MaxPooling2D((2,2)))

    convnet.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    convnet.add(MaxPooling2D((2,2)))

    convnet.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    convnet.add(MaxPooling2D((2,2)))

    # convnet.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    # convnet.add(MaxPooling2D((2,2)))

    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))

    #call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    #layer to merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    #call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = Adam()

    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

    # print('\nsiamese_net summary:')
    # siamese_net.summary()
    
    return siamese_net


def load_wdcnn_net(input_shape = (11001,1),nclasses=10):
    left_input = Input(input_shape)
    convnet = Sequential()
    
    # WDCNN
    convnet.add(Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))


#     print('convnet summary:')
    # convnet.summary()


    encoded_cnn = convnet(left_input)
    prediction_cnn = Dense(nclasses,activation='softmax')(Dropout(0.5)(encoded_cnn ))
    wdcnn_net = Model(inputs=left_input,outputs=prediction_cnn)


    # optimizer = Adam(0.00006)
    optimizer = Adam()
    wdcnn_net.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    # print('\nsiamese_net summary:')
    # cnn_net.summary()
    return wdcnn_net
    
def load_siamese_net_1D(input_shape = (11001,1)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    convnet = Sequential()

    # WDCNN
    convnet.add(Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))
    #convnet.add(Dense(128,activation='relu'))

    #call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    #layer to merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    #call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    # optimizer = Adam(0.00006)
    optimizer = Adam()
    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer) 
    #     print('\nsiamese_net summary:')
    #     siamese_net.summary()
    #     print(siamese_net.count_params())
    return siamese_net