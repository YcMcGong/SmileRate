# Keras import
import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

def load_model(num_classes = 7, type = 'base'):
    
    if type == 'base':
        
        # Start model
        model = Sequential()

        # Model definition

        # CNN 1
        model = Sequential()
        model.add(Conv2D(32, 3, strides = (1,1), padding='valid',
                        input_shape=(48,48,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CNN 2
        model.add(Conv2D(64, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Fully 1
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # # Fully 2
        # model.add(Dense(1024))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # Define optimizer and compile model
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Compile model
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

        print(model.summary())

        return model
        
    elif type == 'all':
        # Start model
        model = Sequential()

        # Model definition

        # CNN 1
        model = Sequential()
        model.add(Conv2D(64, 5, strides = (1,1), padding='valid',
                        input_shape=(48,48,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CNN 2
        model.add(Conv2D(64, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CNN 3
        model.add(Conv2D(128, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # CNN 4
        model.add(Conv2D(128, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CNN 5
        model.add(Conv2D(128, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Fully 1
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Fully 2
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # Define optimizer and compile model
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Compile model
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

        print(model.summary())

        return model

    elif type == 'reduce':

        # Start model
        model = Sequential()

        # Model definition

        # CNN 1
        model = Sequential()
        model.add(Conv2D(64, 5, strides = (1,1), padding='valid',
                        input_shape=(48,48,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CNN 2
        model.add(Conv2D(64, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CNN 3
        model.add(Conv2D(128, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # CNN 4
        model.add(Conv2D(128, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CNN 5
        model.add(Conv2D(128, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Fully 1
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Fully 2
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # Define optimizer and compile model
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Compile model
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

        print(model.summary())

        return model

    elif type == 'good_bad':
        
        # Start model
        model = Sequential()

        # Model definition

        # CNN 1
        model = Sequential()
        model.add(Conv2D(64, 3, strides = (1,1), padding='valid',
                        input_shape=(48,48,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CNN 2
        model.add(Conv2D(64, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CNN 3
        model.add(Conv2D(128, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # CNN 4
        model.add(Conv2D(128, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CNN 5
        model.add(Conv2D(128, 3, strides = (1,1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Fully 1
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Fully 2
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # Define optimizer and compile model
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Compile model
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

        print(model.summary())

        return model

    else:

        return 0