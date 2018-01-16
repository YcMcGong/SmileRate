# Training script
from data_feeder import data_feeder
from matplotlib import pyplot as plt
import numpy as np
# Keras import
import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
# Model import
from model import load_model

# Define path for saved model
MODEL_PATH = 'models/weights.CNN_2_4_4_32_layer'

# Using data feeder to load data
data_feeder = data_feeder()
data_feeder.load_data()
test_X = data_feeder.test_X
test_Y = data_feeder.test_Y

# Start model
model = Sequential()
batch_size = 2000
num_classes = 7
epochs = 1

# Load model
model = load_model()

# Set checkpoint
path = MODEL_PATH
verbose = 1 # Set to show data
checkpointer = ModelCheckpoint(filepath=path, 
                                        verbose=verbose, save_best_only=True)

# Load the best trained model
model.load_weights(path)

# Evaluate model
evaluate_batch_size = batch_size
result = model.evaluate(x=test_X, y=test_Y, batch_size=evaluate_batch_size, verbose=1)
print('loss:     ', result[0])
print('accuracy: ', result[1])