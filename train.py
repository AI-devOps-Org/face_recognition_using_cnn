import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = './faces'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to get the images and label data
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for imagePath in image_paths:
        # Convert it to grayscale
        pil_image = Image.open(imagePath).convert('L')
        img_numpy = np.array(pil_image, 'uint8')

        # Extract the face ID from the image file name
        id = int(os.path.split(imagePath)[-1].split("user")[1].split(".")[0])
        
        # Detect the face in the image
        faces = detector.detectMultiScale(img_numpy)
        
        # If a face is detected, append it to the training data
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return face_samples, ids

print("Training faces. It will take a few seconds. Wait ...")
faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
recognizer.write('trainer.xml')

print(f"{len(np.unique(ids))} faces trained. Exiting Program")
