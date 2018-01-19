# This file create a set of 48 by 48 human face images from a single image
import matplotlib.pyplot as plt
import numpy as np         
import cv2

# Path for pre-trained face detector
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# Face size
FACE_SIZE = 48

def get_faces(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    # find faces in image
    found_faces = face_cascade.detectMultiScale(gray)

    # Get cropped and resized faces array
    faces = []
    for (x,y,w,h) in found_faces:
        
        # Crop and resize
        face = cv2.resize(gray[y:y+h, x:x+w], (FACE_SIZE,FACE_SIZE))
        face = face.reshape(1, FACE_SIZE, FACE_SIZE, 1)/255.0
        faces.append(face)

        # add bounding box to color image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # display the image, along with bounding box
    plt.imshow(cv_rgb)
    plt.show()

    # self.faces = np.array(self.faces).reshape(-1,FACE_SIZE, FACE_SIZE, 1)
    
    # Return faces array for ml
    return faces


if __name__ == '__main__':

    PATH = 'photo/test.jpg'
    # load color (BGR) image
    img = cv2.imread(PATH)
    faces = get_faces(img)
    print(faces)
