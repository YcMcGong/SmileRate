# This file analyze the faces in a photo
import matplotlib.pyplot as plt
from model import load_model
import numpy as np
import face_detector
import cv2

FACE_SIZE = 48

class face_analyzer():

    def __init__(self):
        
        # Load models
        # self.model = load_model(type = 'all')
        # self.model.load_weights('models/weights.CNN_complex_layer_best')

        self.model = load_model(num_classes = 3, type = 'good_bad')
        self.model.load_weights('models/weights.CNN_new_good_bad')

    def analyze_single_photo(self, img):

        faces = face_detector.get_faces(img)

        for face in faces:
            plt.imshow(face.reshape(48,48))
            plt.show()
            print(self.model.predict(face))

    def analyze_many_photo(self, path):
        
        for i in range(1,14):
            img = cv2.imread(path + str(i) + '.jpg')
            faces = face_detector.get_faces(img)

            for face in faces:
                plt.imshow(face.reshape(48,48))
                plt.show()
                print(self.model.predict(face))

    def rate_a_photo(self, img):
        
        faces = face_detector.get_faces(img)
        faces = np.array(faces).reshape(-1, FACE_SIZE, FACE_SIZE, 1)
        results = self.__predict_faces(faces)
        rating_list = self.__get_rating_from_result(results)
        rate = sum(rating_list)

        return rate

    def __predict_faces(self, faces):
        
        results = self.model.predict(faces)
        return results
        

    def __get_rating_from_result(self, results):
        # Face format [ [ good  bad  neutral] ]
        rating_list = []
        for result in results:
            good = result[0]
            bad = result[1]
            neutral = result[0]
            rate = good - bad - 0.1*neutral
            rating_list.append(rate)

        return rating_list
            

if __name__ == '__main__':
    
    img = cv2.imread('data/photos/13.jpg')
    face_analyzer = face_analyzer()
    print(face_analyzer.rate_a_photo(img))
    # face_analyzer.analyze_single_photo(img)
    # face_analyzer.analyze_many_photo('data/photos/')

            
            
