import cv2 as cv
import numpy as np

# Load the Haar cascade for face detection
haar_cascade = cv.CascadeClassifier("haar_face.xml")

# Load the trained LBPH face recognizer model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

# Load the list of person names (in the same order as training labels)
people = list(np.load('people.npy'))
print("People:", people)

# Load the image to validate/predict
imagePath = r"Validate\Lionel Messi\1.jpg"
img = cv.imread(imagePath)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert image to grayscale
cv.imshow('Input', img)

# Detect faces in the image using Haar cascade
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Iterate over all detected faces
for (x, y, w, h) in faces_rect:
    # Extract region of interest (face area)
    roi = gray[y:y+h, x:x+w]

    # Predict the label and confidence score using the recognizer
    label, confidence = face_recognizer.predict(roi)
    print(f"Prediction: {people[label]} | Confidence: {confidence:.2f}")

    # Decide the text to display
    text = people[label]

    # Calculate text width for horizontal centering
    font = cv.FONT_HERSHEY_COMPLEX
    (text_width, _), _ = cv.getTextSize(text, font, 1.0, 2)
    text_x = (img.shape[1] - text_width) // 2

    # Draw the prediction label above the detected face
    cv.putText(img, text, (text_x, y - 10), font, 1.0, (0, 255, 0), 2)

    # Draw a rectangle around the detected face
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the final image with prediction
cv.imshow('Result', img)
cv.waitKey(0)
cv.destroyAllWindows()
