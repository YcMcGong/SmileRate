# This file handle data feeding for the system.abs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import os.path

# Init data path
CSV_DATA_PATH = 'data/data.csv'
DATA_PATH = 'data/'
IMG_SIZE = 48
NUMBER_OF_CLASS = 6

class data_feeder():

    def __init__(self):
        
        self.data_path = None
        self.data = None
        self.train_X = None
        self.test_Y = None
        self.train_Y = None
        self.test_X = None

    def load_data(self, path = CSV_DATA_PATH):

        self.data_path = path

        if (os.path.exists(DATA_PATH + 'train_X') and os.path.exists(DATA_PATH + 'test_X') 
        and os.path.exists(DATA_PATH + 'train_Y') and os.path.exists(DATA_PATH + 'test_Y')
        and os.path.exists(DATA_PATH + 'data')):

            self.data = joblib.load(DATA_PATH+'data')
            self.train_X = joblib.load(DATA_PATH+'train_X')
            self.train_Y = joblib.load(DATA_PATH+'train_Y')
            self.test_X = joblib.load(DATA_PATH+'test_X')
            self.test_Y = joblib.load(DATA_PATH+'test_Y')
        
        else:
            
            df = pd.read_csv(self.data_path)
            self.data = df
            joblib.dump(self.data, DATA_PATH+'data')

            # Load training data
            df_train = df.loc[df['Usage'] == 'Training']
            train_df_X = df_train['pixels'].values
            self.train_X = self.str_list_to_int_list(train_df_X)
            self.train_Y = np.array(df_train['emotion'].values).reshape(-1,1)
            enc = OneHotEncoder()
            self.train_Y = enc.fit_transform(self.train_Y).toarray()
            joblib.dump(self.train_X, DATA_PATH+'train_X') # Save the processed data
            joblib.dump(self.train_Y, DATA_PATH+'train_Y')

            # Load testing data
            df_test = df.loc[df['Usage'] == 'PrivateTest']
            test_df_X = df_test['pixels'].values
            self.test_X = self.str_list_to_int_list(test_df_X)
            self.test_Y = np.array(df_test['emotion'].values).reshape(-1,1)
            enc = OneHotEncoder()
            self.test_Y = enc.fit_transform(self.test_Y).toarray()
            joblib.dump(self.test_X, DATA_PATH+'test_X') # Save the processed data
            joblib.dump(self.test_Y, DATA_PATH+'test_Y')


    def str_list_to_int_list(self, df):
        
        data_X_set = []
        for image in df:
            pixels = np.array([int(pixel) for pixel in image.split()]).reshape(IMG_SIZE, IMG_SIZE, 1)
            data_X_set.append(pixels)
        
        # normalize inputs from 0-255 to 0.0-1.0
        data_X_set = np.array(data_X_set).astype('float32') / 255.0
        return data_X_set

    def train_validation_split(self, validation_ratio = 0.1):
        self.train_X_train, self.train_X_validation, self.train_Y_train, self.train_Y_validation = train_test_split(
            self.train_X, self.train_Y, test_size=validation_ratio, random_state=35)

if __name__ == '__main__':
    data_feeder = data_feeder()
    data_feeder.load_data(CSV_DATA_PATH)
    print(data_feeder.test_Y)
    # print(data_feeder.test_Y[0,1])

        