import os
import cv2 as cv
import numpy as np

# Path to training dataset folder
DIR = r"Train"

# Load Haar cascade classifier for face detection
haar_cascade = cv.CascadeClassifier("haar_face.xml")

# Automatically generate a sorted list of person names (from folder names)
people = sorted(os.listdir(DIR))
print("People:", people)

# Initialize feature and label lists
features = []  # List of face regions (grayscale)
labels = []    # Corresponding label index for each face

# -----------------------------
# Function to prepare training data
# -----------------------------
def create_train():
    for person in people:
        label = people.index(person)  # Assign a numeric label based on order
        path = os.path.join(DIR, person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            # Read image
            img_array = cv.imread(img_path)
            if img_array is None:
                continue  # Skip if unreadable or invalid

            # Convert image to grayscale (required for Haar detection)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # Extract and store each face ROI
            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w]
                features.append(face_roi)
                labels.append(label)

# Call the training data preparation function
create_train()
print("Training Done")

# Convert feature and label lists into NumPy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# -----------------------------
# Train the LBPH Face Recognizer
# -----------------------------
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

# Save the trained model and data for later use
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
np.save('people.npy', np.array(people))
