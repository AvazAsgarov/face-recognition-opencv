# face-recognition-opencv

Face recognition system using Haar Cascade Classifier for face detection and LBPH (Local Binary Pattern Histogram) for face recognition — built with OpenCV and Python.

---

# Tools & Technologies

- Python 3.10+
- OpenCV (`opencv-contrib-python`)
- NumPy
- Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)
- LBPH Face Recognizer (OpenCV)

---

# Methodology

- **Face Detection**: Uses Haar Cascades to locate faces in grayscale images.
- **Face Recognition**: Extracts face ROIs, trains LBPH model with labeled samples.
- **Evaluation**: Predicts test images and draws labels using confidence scores.
- **Label Mapping**: Class labels are derived from folder names (person-wise).

---

# Project Structure

Train/          # Training images (organized in folders per person)
Validate/       # Test images
train_faces.py  # Training script
recognize_faces.py  # Prediction script
face_trained.yml  # Saved model
features.npy / labels.npy / people.npy
haar_face.xml   # Haar cascade file
