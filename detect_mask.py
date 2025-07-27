import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model using full path
model = load_model("C:/Users/moort/Downloads/mask_detector_model.h5")

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Preprocess face image
        resized = cv2.resize(face, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 3))

        # Predict mask/no mask
        result = model.predict(reshaped)
        label = "Mask" if result[0][0] < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the output
    cv2.imshow("Face Mask Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and destroy window
cap.release()
cv2.destroyAllWindows()
if label == "No Mask":
    color = (0, 0, 255)  # Red for alert
    label_display = "No Mask"
    
    # 🔊 Play beep sound
    winsound.Beep(1000, 500)  # frequency=1000Hz, duration=500ms
