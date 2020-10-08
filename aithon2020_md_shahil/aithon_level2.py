#!/usr/bin/env python



import sys, os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/source/')


import source.classification  as cls




'''
The following function will be called to train and test your model.
The function name, signature and output type is fixed.
The first argument is file name that contain data for training.
The second argument is file name that contain data for test.
The function must return predicted values or emotion for each data in test dataset
sequentially in a list.
['sad', 'happy', 'fear', 'fear', ... , 'happy']
'''

def  aithon_level2_api(traincsv, testcsv):

    # The following dummy code for demonstration.

    # Train the model with training data
    cls.train_a_model(traincsv)

    # Test that model with test data
    # And return predicted emotions in a list
    return cls.test_the_model(testcsv)

print('please pass the Train and test in csv format')
print('Name your trainig dataset Eg:- data/aithon2020_level2_traning.csv')
traincsv=input()

print('Name your test dataset Eg:- data/aithon2020_level2_testdata.csv')
testcsv=input()

aithon_level2_api(traincsv=traincsv, testcsv=testcsv)
