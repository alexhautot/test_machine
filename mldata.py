import numpy as np

import h5py

def get_training_testing_data(train, test):
    # Loading the data files
    train_file = h5py.File(train, 'r')
    test_file = h5py.File(test, 'r')
    x_train = train_file['train_set_x'].value
    y_train = train_file['train_set_y'].value
    x_test = test_file['test_set_x'].value
    y_test = test_file['test_set_y'].value

    train_file.close()
    test_file.close()

    #print('x_train.shape:', x_train.shape)
    #print('y_train.shape:', y_train.shape)
    #print('x_test.shape:', x_test.shape)
    #print('y_test.shape:', y_test.shape)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
    y_train = y_train.reshape(y_train.shape[0], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
    y_test = y_test.reshape(y_test.shape[0], 1)

    x_train = x_train/255
    x_test = x_test/255

    return x_train, y_train, x_test, y_test
