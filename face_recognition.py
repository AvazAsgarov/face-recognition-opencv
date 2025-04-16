import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier("haar_face.xml")
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

people = list(np.load('people.npy'))
print("People:", people)

imagePath = r"Validate\Lionel Messi\1.jpg"
img = cv.imread(imagePath)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Input', img)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in faces_rect:
    roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(roi)

    print(f"Prediction: {people[label]} | Confidence: {confidence:.2f}")

    # Centered label
    text = people[label]
    font = cv.FONT_HERSHEY_COMPLEX
    (text_width, _), _ = cv.getTextSize(text, font, 1.0, 2)
    text_x = (img.shape[1] - text_width) // 2
    cv.putText(img, text, (text_x, y - 10), font, 1.0, (0, 255, 0), 2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv.imshow('Result', img)
cv.waitKey(0)
cv.destroyAllWindows()