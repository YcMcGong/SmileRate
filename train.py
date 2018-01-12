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

# Define path for saved model
MODEL_PATH = 'models/weights.CNN_2_layer'

# Using data feeder to load data
data_feeder = data_feeder()
data_feeder.load_data()
data_feeder.train_validation_split()
train_X = data_feeder.train_X_train
train_Y = data_feeder.train_Y_train
train_X_validation = data_feeder.train_X_validation
train_Y_validation = data_feeder.train_Y_validation
test_X = data_feeder.test_X
test_Y = data_feeder.test_Y

# Start model
model = Sequential()
batch_size = 2000
num_classes = 7
epochs = 1

# Model definition
model = Sequential()
model.add(Conv2D(32, 3, strides = (1,1), padding='valid',
                 input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Define optimizer and compile model
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print(model.summary())

# Set checkpoint
path = MODEL_PATH
verbose = 1 # Set to show data
checkpointer = ModelCheckpoint(filepath=path, 
                                        verbose=verbose, save_best_only=True)

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
