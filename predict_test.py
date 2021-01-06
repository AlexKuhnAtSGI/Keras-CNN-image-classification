import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

def train_accuracy():
	sample_height = 64
	sample_width = 128
	sample_channels = 1

	df = pd.read_pickle("Train.pkl")
	df = df.reshape(60000,-1).astype("float32")
	x_train = df/255

	x_train = x_train.reshape(x_train.shape[0], sample_height, sample_width, sample_channels)

	y_train = pd.read_csv("TrainLabels.csv", header=None).to_numpy()
	y_train = keras.utils.to_categorical(y_train)

	model = keras.models.load_model('best_model.hdf5')
	model.evaluate(x_train, y_train)
	
def test_predictions():
	sample_height = 64
	sample_width = 128
	sample_channels = 1
	
	df = pd.read_pickle("Test.pkl")
	print(df.shape)
	df = df.reshape(10000,-1).astype("float32")
	x_test = df/255
	x_test = x_test.reshape(x_test.shape[0], sample_height, sample_width, sample_channels)
	
	model = keras.models.load_model('best_model.hdf5')
	pred = np.vstack(model.predict_classes(x_test))
	ids = np.vstack(list(range(pred.size)))
	# print(pred[0:5])
	np.savetxt('TestLabels_2.csv', np.column_stack((ids,pred)),header="id,output", delimiter=',', fmt="%i,%i")

train_accuracy()	
test_predictions()