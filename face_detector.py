# This file create a set of 48 by 48 human face images from a single image
import matplotlib.pyplot as plt
import numpy as np         
import cv2

# Path for pre-trained face detector
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# Face size
FACE_SIZE = 48

class face_feeder():

    def __init__(self, img):
        self.img = img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_faces(self):
        
        # extract pre-trained face detector
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

        # find faces in image
        found_faces = face_cascade.detectMultiScale(self.gray)

        # Get cropped and resized faces array
        self.faces = []
        for (x,y,w,h) in found_faces:
            # Crop and resize
            face = cv2.resize(self.gray[y:y+h, x:x+w], (FACE_SIZE,FACE_SIZE))
            self.faces.append(face)

        self.faces = np.array(self.faces).reshape(-1,FACE_SIZE, FACE_SIZE, 1)
        
        # Return faces array for ml
        return self.faces


if __name__ == '__main__':

    PATH = 'photo/test.jpg'
    # load color (BGR) image
    img = cv2.imread(PATH)

    face_feeder = face_feeder(img)
    faces = face_feeder.get_faces()
    print(faces)
