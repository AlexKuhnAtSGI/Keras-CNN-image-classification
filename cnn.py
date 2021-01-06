import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
# from sklearn.svm import SVC

def create_model():
	#First attempt at the model, with one convolutional layer and a 4x4 filter size
	#Did not do well
	model = keras.Sequential()
	model.add(keras.layers.Conv2D(64, (4, 4), activation='relu', kernel_initializer='he_uniform', input_shape=(sample_height, sample_width, sample_channels)))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(keras.layers.Dense(10, activation='softmax'))
	return model
	
def create_model2():
	#Second model attempt, adding dropout and an input layer
	#Slightly better, but clearly not good enough
	#both models were overfitting the data: achieving 99%+ training accuracy and getting no higher than 50% on the validation set
	model = keras.Sequential()
	model.add(keras.layers.InputLayer(input_shape=(sample_height,sample_width,sample_channels)))
	model.add(keras.layers.Conv2D(64, (4, 4), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Dropout(0.1))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(keras.layers.Dense(10, activation='softmax'))
	return model
	
def create_model_final():
	#Final version of the 3rd model we tried, which ended up working out the best
	#Features 4 convolutional layers, batch normalization and dropout at every layer to prevent overfitting
	#5x5 filter size was used (originally had 3x3 but that was too focused on fine details)
	model = keras.Sequential()
	model.add(keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=(sample_height, sample_width, sample_channels)))
	model.add(keras.layers.BatchNormalization())

	model.add(keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Dropout(0.25))

	model.add(keras.layers.Conv2D(64, kernel_size=(7, 7), activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dropout(0.25))

	model.add(keras.layers.Conv2D(128, kernel_size=(7, 7), activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Dropout(0.25))

	model.add(keras.layers.Flatten())

	model.add(keras.layers.Dense(512, activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dropout(0.5))

	model.add(keras.layers.Dense(128, activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dropout(0.5))

	model.add(keras.layers.Dense(10, activation='softmax'))
	
	return model
	
def create_model4():
	#This was an attempt with 2 convolutional layers to try and speed things up
	#Unfortunately, the loss of accuracy was too great
	model = keras.Sequential()
	model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(sample_height, sample_width, sample_channels)))
	model.add(keras.layers.Activation("relu"))
	model.add(keras.layers.BatchNormalization(axis=-1))
	model.add(keras.layers.Conv2D(32, (3, 3), padding="same"))
	model.add(keras.layers.Activation("relu"))
	model.add(keras.layers.BatchNormalization(axis=-1))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Dropout(0.25))
	
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(512))
	model.add(keras.layers.Activation("relu"))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Dropout(0.5))
	
	model.add(keras.layers.Dense(10))
	model.add(keras.layers.Activation("softmax"))
	
	return model

df = pd.read_pickle("Train.pkl")
df = df.reshape(60000,-1).astype("float32")

#Preprocessing step here: normalizing each pixel value (i.e., each feature) to be between 0 and 1 rather than 0 and 255
x_train = df/255

#Image dimensions are 64*128, so we need to recreate that for the CNN
sample_height = 64
sample_width = 128
sample_channels = 1
x_train = x_train.reshape(x_train.shape[0], sample_height, sample_width, sample_channels)

#Using one-hot encoding for the y-values so that the classifier's predictions are more accurate
y_train = pd.read_csv("TrainLabels.csv", header=None).to_numpy()
y_train = keras.utils.to_categorical(y_train)

#Optimizer is keras' Adam, a variant of stochastic gradient descent that computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients
opt = keras.optimizers.Adam()

#Using a keras callback to preserve the best model in terms of validation loss over all our 20 epochs per fold
mc = keras.callbacks.ModelCheckpoint(filepath="best_model.hdf5", monitor='val_loss', save_best_only=True)

kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(x_train, y_train):
	xTrn, xTes = x_train[train_index], x_train[test_index]
	yTrn, yTes = y_train[train_index], y_train[test_index]
	
	model = create_model_final()
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(xTrn, yTrn, epochs=20,batch_size=64, validation_data=(xTes,yTes), callbacks=[mc])