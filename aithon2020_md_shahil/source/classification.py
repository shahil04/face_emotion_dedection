import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import sklearn
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



'''
The following dummy code for demonstration.
'''


def train_a_model(trainfile):
    '''
    :param trainfile:
    :return:
    '''
    #load the dataset using pandas
    data=pd.read_csv(trainfile)
    #check len of rows and its intance len

    with open(trainfile) as f:
        content = f.readlines()

    lines = np.array(content)

    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    
    #save the data in train and validation data
    x_train, y_train = [], []

    for i in range(1,num_of_instances):
        try:
            emotion = lines[i].split(",")[0]

            val = lines[i].split(",")[1:]

            pixels = np.array(val, 'float32')

            y_train.append(emotion)
            x_train.append(pixels)
        except:
            print("",end="")
            
            
     # data transformation for train and test sets
    x_train = np.array(x_train, 'float32')
    
    x_train /= 255 #normalize inputs between [0, 1]
    
    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype('float32')
   
    print(x_train.shape[0], 'train samples')
    
    y_train= []

    for i in range(1,num_of_instances):
        try:
            emotion = lines[i].split(",")[0]



            y_train.append(emotion)
        except:
            print("",end="")

    le=LabelEncoder()
    y_train=le.fit_transform(y_train)
    a_train=[]
    for i in range(1,num_of_instances):
        emotion = y_train[i-1]
        emotion = keras.utils.to_categorical(emotion, 3)

        a_train.append(emotion)
    
    y_train=a_train
    y_train = np.array(y_train, 'float32')
    
    #variables
    num_classes = 3
    #fear, happy, sad
    batch_size = 256
    epochs = 10
    
    
    #construct CNN structure
    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    #2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    #3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    model.add(Flatten())

    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    #-----------------------------
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    #-----------------------------

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    ################################
    #------------------------------
    #batch process
    gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

    #------------------------------

    model.compile(loss='categorical_crossentropy'
        , optimizer=keras.optimizers.Adam()
        , metrics=['accuracy']
    )

    #------------------------------
    

    print('Training start please wait....')
    #model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset
    model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs) #train for randomly selected one

    print("model Training completed")
    model.save('models.h5')
    
    return model


def test_the_model(testfile):
    '''

    :param testfile:
    :return:  a list of predicted values in same order of
    '''
    
    #load the dataset using pandas
    data=pd.read_csv(testfile)
    #check len of rows and its intance len

    with open(testfile) as f:
        content = f.readlines()

    lines = np.array(content)

    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
   
    #save the data in train and validation data
    x_test, y_test = [], []

    for i in range(1,num_of_instances):
        try:
            emotion = lines[i].split(",")[0]

            val = lines[i].split(",")[1:]

            pixels = np.array(val, 'float32')

            y_test.append(emotion)
            x_test.append(pixels)
        except:
            print("",end="")
            
            
     # data transformation for train and test sets
    x_test = np.array(x_test, 'float32')

    x_test /= 255 #normalize inputs between [0, 1]
   

    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype('float32')
   
    print(x_test.shape[0], 'test samples')
    
    
    y_test= []

    for i in range(1,num_of_instances):
        try:
            emotion = lines[i].split(",")[0]

            y_test.append(emotion)
        except:
            print("",end="")

    #Now predict the test data by train model
    model=keras.models.load_model('models.h5')
    pred=model.predict(x_test)
    
    result=[]
    for i in pred:
        for j,k in enumerate(i):
            objects = ('Fear', 'Happy', 'Sad')
            if k==i.max():
                print(objects[j])
                result.append(objects[j])
    
    return result

 
    
    
