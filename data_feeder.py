# This file handle data feeding for the system.abs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

# Init data path
CSV_DATA_PATH = 'data/data.csv'
DATA_PATH = 'data/'
IMG_SIZE = 48
NUMBER_OF_CLASS = 7

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
    
    def train_validation_split_subset(self, target, validation_ratio = 0.1): # target is a list

        # Check the original data to get a list of emotions and re-arrange the train and test data
        # Load training data
        df_train = self.data.loc[self.data['Usage'] == 'Training']
        emotion_order_train = df_train['emotion']

        # Load testing data
        df_test = self.data.loc[self.data['Usage'] == 'PrivateTest']
        emotion_order_test = df_test['emotion']

        # Re-order training set to create subset
        new_train_X_subset = []
        new_train_Y_subset = []
        new_test_X_subset = []
        new_test_Y_subset = []
        for index, emotion in enumerate(emotion_order_train):
            if emotion in target:
                new_train_X_subset.append(self.train_X[index])
                new_train_Y_subset.append(emotion)
        for index, emotion in enumerate(emotion_order_test):
            if emotion in target:
                new_test_X_subset.append(self.test_X[index])
                new_test_Y_subset.append(emotion)

        train_X_subset = np.array(new_train_X_subset) 
        train_Y_subset = np.array(new_train_Y_subset).reshape(-1,1)
        self.test_X_subset = np.array(new_test_X_subset)
        test_Y_subset = np.array(new_test_Y_subset).reshape(-1,1)

        enc = OneHotEncoder()
        train_Y_subset = enc.fit_transform(train_Y_subset).toarray()
        self.test_Y_subset = enc.fit_transform(test_Y_subset).toarray()

        self.train_X_subset_train, self.train_X_subset_validation, self.train_Y_subset_train, self.train_Y_subset_validation = train_test_split(
            train_X_subset, train_Y_subset, test_size=validation_ratio, random_state=35)


    def train_validation_split_good_bad_neutral_split(self, good = [3,5], bad = [0, 1, 2 , 4], neutral = [6], validation_ratio = 0.1): # target is a list

        # Check the original data to get a list of emotions and re-arrange the train and test data
        # Load training data
        df_train = self.data.loc[self.data['Usage'] == 'Training']
        emotion_order_train = df_train['emotion']

        # Load testing data
        df_test = self.data.loc[self.data['Usage'] == 'PrivateTest']
        emotion_order_test = df_test['emotion']

        # Re-order training set to create subset
        new_train_X_subset = []
        new_train_Y_subset = []
        new_test_X_subset = []
        new_test_Y_subset = []
        for index, emotion in enumerate(emotion_order_train):
            if emotion in good:
                new_train_X_subset.append(self.train_X[index])
                new_train_Y_subset.append(0)
            
            elif emotion in bad:
                new_train_X_subset.append(self.train_X[index])
                new_train_Y_subset.append(1)

            elif emotion in neutral:
                new_train_X_subset.append(self.train_X[index])
                new_train_Y_subset.append(2)
                
        for index, emotion in enumerate(emotion_order_test):
            if emotion in good:
                new_test_X_subset.append(self.test_X[index])
                new_test_Y_subset.append(0)
            
            elif emotion in bad:
                new_test_X_subset.append(self.test_X[index])
                new_test_Y_subset.append(1)

            elif emotion in neutral:
                new_test_X_subset.append(self.test_X[index])
                new_test_Y_subset.append(2)

        train_X_subset = np.array(new_train_X_subset) 
        train_Y_subset = np.array(new_train_Y_subset).reshape(-1,1)
        self.test_X_subset = np.array(new_test_X_subset)
        test_Y_subset = np.array(new_test_Y_subset).reshape(-1,1)

        enc = OneHotEncoder()
        train_Y_subset = enc.fit_transform(train_Y_subset).toarray()
        self.test_Y_subset = enc.fit_transform(test_Y_subset).toarray()

        self.train_X_subset_train, self.train_X_subset_validation, self.train_Y_subset_train, self.train_Y_subset_validation = train_test_split(
            train_X_subset, train_Y_subset, test_size=validation_ratio, random_state=35)

    def print_distribution(self):
        
        self.class_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        # Init the list to store statistic
        self.training_statistic = []
        self.testing_statistic = []

        # Logics
        df = self.data
        df_train = df.loc[df['Usage'] == 'Training']
        df_test = df.loc[df['Usage'] == 'PrivateTest']
        train_Y = df_train['emotion'].values
        test_Y = df_test['emotion'].values

        print('Training Statistic \n')
        for i in range(NUMBER_OF_CLASS):
            class_name = self.class_list[i]
            number_of_sample = sum(train_Y==i)
            print(class_name, number_of_sample)
            self.training_statistic.append(number_of_sample)

        print('\nTesting Statistic \n')
        for i in range(NUMBER_OF_CLASS):
            class_name = self.class_list[i]
            number_of_sample = sum(test_Y==i)
            print(class_name, number_of_sample)
            self.testing_statistic.append(number_of_sample)

    def plot_data_distribution(self):
        
        index = np.arange(7)
        bar_width = 0.35       
        opacity = 0.4    

        plt.bar(index, self.training_statistic, bar_width, alpha=opacity, color ='b', label = 'Train')
        # plt.figure()
        plt.bar(index + bar_width, self.testing_statistic, bar_width, alpha=opacity, color ='r', label = 'Test')

        plt.xlabel('Emotion')
        plt.ylabel('Number of Sample')    
        plt.title('Emotion Sample Distribution')    
        plt.xticks(index + bar_width, self.class_list)

        plt.legend()
        plt.tight_layout()

        plt.show()
        
        
        


if __name__ == '__main__':
    data_feeder = data_feeder()
    data_feeder.load_data(CSV_DATA_PATH)
    data_feeder.train_validation_split_subset(target = [0,2])