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
MODEL_PATH = 'models/weights.CNN_new'

# Using data feeder to load data

# All data
# data_feeder = data_feeder()
# data_feeder.load_data()
# data_feeder.train_validation_split()
# train_X = data_feeder.train_X_train
# train_Y = data_feeder.train_Y_train
# train_X_validation = data_feeder.train_X_validation
# train_Y_validation = data_feeder.train_Y_validation
# test_X = data_feeder.test_X
# test_Y = data_feeder.test_Y

# Subset data
# data_feeder = data_feeder()
# data_feeder.load_data()
# data_feeder.train_validation_split_subset(target = [0,2,3,4,5,6])
# train_X = data_feeder.train_X_subset_train
# train_Y = data_feeder.train_Y_subset_train
# train_X_validation = data_feeder.train_X_subset_validation
# train_Y_validation = data_feeder.train_Y_subset_validation
# test_X = data_feeder.test_X_subset
# test_Y = data_feeder.test_Y_subset


# Good bad data
data_feeder = data_feeder()
data_feeder.load_data()
data_feeder.train_validation_split_good_bad_neutral_split()
train_X = data_feeder.train_X_subset_train
train_Y = data_feeder.train_Y_subset_train
train_X_validation = data_feeder.train_X_subset_validation
train_Y_validation = data_feeder.train_Y_subset_validation
test_X = data_feeder.test_X_subset
test_Y = data_feeder.test_Y_subset

# Load parameters
batch_size = 500
num_classes = 3
epochs = 500

# Load model
model = load_model(num_classes)

# Set checkpoint
path = MODEL_PATH
verbose = 1 # Set to show data
checkpointer = ModelCheckpoint(filepath=path, 
                                        verbose=verbose, save_best_only=True)

# Try loading the best trained model from last
try:
    model.load_weights(path)
except:
    print('new model created')

# Train model
model.fit(train_X, train_Y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(train_X_validation, train_Y_validation),
            shuffle=True, callbacks=[checkpointer], verbose = verbose)

# Load the best trained model
model.load_weights(path)

# Evaluate model
evaluate_batch_size = batch_size
result = model.evaluate(x=test_X, y=test_Y, batch_size=evaluate_batch_size, verbose=1)
print('loss:     ', result[0])
print('accuracy: ', result[1])