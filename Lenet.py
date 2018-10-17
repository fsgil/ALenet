import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
import os

#load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# unsure if some preprocessing of data is necessary

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

###############################################################################
# create model for the network
model = Sequential()

# layer one
model.add( Convolution2D( filters=20, kernel_size=(5,5), padding='same', input_shape=(32,32,3)) )

# activation function for layer one
model.add( Activation( activation = 'relu') )

# pooling layer two
model.add(MaxPooling2D(pool_size=(2,2) , strides=(2,2) ) )

# layer three
model.add( Convolution2D( filters=20, kernel_size=(5,5), padding='same', input_shape=(32,32,3)) )

# activation function for layer three
model.add( Activation( activation = 'relu') )

# pooling layer four
model.add( MaxPooling2D(pool_size=(2,2) , strides=(2,2) ) )

# Flatten the network
model.add( Flatten() )

# fully connected layer 5
model.add( Dense(500) )

# activation layer for layer 5
model.add( Activation( activation = "relu") )

# fully conneceted layer 6
model.add( Dense(10) )

# Activation function of layer 6
model.add( Activation("softmax") )

###############################################################################
# train should be here


# Compile the network
model.compile( loss = "categorical_crossentropy",  optimizer = SGD(lr = 0.001, momentum=0.9), metrics = ["accuracy"] )

# Train the model 
model.fit( x_train, y_train, batch_size = 128, nb_epoch = 25, verbose = 1 )

# Evaluate the model
(loss, accuracy) = model.evaluate( x_test, y_test, batch_size = 128, verbose = 1 )

# Print the model's accuracy
print(accuracy)
print(loss)