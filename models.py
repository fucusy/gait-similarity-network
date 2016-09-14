from keras.layers import Merge, Convolution2D
from keras.layers import Dense, Activation, MaxPooling2D, MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.convolutional import ZeroPadding2D

def merger_test_model():
    left_branch = Sequential()
    left_branch.add(Dense(32, input_dim=784))
    
    right_branch = Sequential()
    right_branch.add(Dense(32, input_dim=784))
    
    merged = Merge([left_branch, right_branch], mode='concat')
    
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(10, activation='softmax'))

    final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # you can train the model by below example code
    # final_model.fit([input_data_1, input_data_2], targets)  
    # we pass one data array per model input

    return final_model

def cas_mt_model(lr, weight_path):
    row = 210
    col = 70
    left_branch = Sequential()
    left_branch.add(Convolution2D(16, 7, 7, border_mode='valid', input_shape=(1, row, col)))
    left_branch.add(Activation('relu'))
    left_branch.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

    left_branch.add(Convolution2D(4, 7, 7, border_mode='valid', input_shape=(16, 107, 32)))
    left_branch.add(Activation('relu'))
    left_branch.add(ZeroPadding2D(padding=(1, 0)))
    left_branch.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

    # how to add normalization layer 
    right_branch = Sequential()
    right_branch.add(Convolution2D(16, 7, 7, border_mode='valid', input_shape=(1, row, col)))
    right_branch.add(Activation('relu'))
    right_branch.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

    right_branch.add(Convolution2D(4, 7, 7, border_mode='valid', input_shape=(16, 107, 32)))
    right_branch.add(Activation('relu'))
    right_branch.add(ZeroPadding2D(padding=(1, 0)))
    right_branch.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    
    
    merged = Merge([left_branch, right_branch], mode=lambda x: abs(x[0] - x[1]), output_shape=(64, 51, 13))
    
    final_model = Sequential()
    final_model.add(merged)
    #final_model.add(Convolution2D(4, 7, 7, border_mode='valid', input_shape=(64, 51, 13)))
    final_model.add(Flatten())
    final_model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    final_model.compile(optimizer=sgd, loss='categorical_crossentropy')

    # you can train the model by below example code
    # final_model.fit([input_data_1, input_data_2], targets)  
    # we pass one data array per model input

    return final_model
    

if __name__ == "__main__":
    lr=1e-4
    weight_path=""
    model = cas_mt_model(lr, weight_path)
    # generate dummy data
    import numpy as np
    from keras.utils.np_utils import to_categorical
    data_1 = np.random.random((1000, 1, 210, 70))
    data_2 = np.random.random((1000, 1, 210, 70))
    
    # these are integers between 0 and 9
    labels = np.random.randint(2, size=(1000, 1))
    # we convert the labels to a binary matrix of size (1000, 10)
    # for use with categorical_crossentropy
    labels = to_categorical(labels, 2)
    
    # train the model
    # note that we are passing a list of Numpy arrays as training data
    # since the model has 2 inputs
    model.fit([data_1, data_2], labels, nb_epoch=10, batch_size=32)   
