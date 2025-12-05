import cv2
import numpy as np
from collections import deque

# Load labels and face buffer
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_buffer = deque(maxlen=10)

# Load DNN face detector
prototxt_path = 'X:/emotion detection/app/deploy.prototxt.txt'
weights_path = 'X:/emotion detection/app/res10_300x300_ssd_iter_140000.caffemodel'
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

def detect_and_predict_face_emotion(frame, emotion_model):
    h, w = frame.shape[:2]
    
    # Prepare input blob for DNN
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    emotion = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # You can adjust threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Ensure bounding box is within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))
            gray = gray.astype("float32") / 255.0
            gray = np.expand_dims(gray, axis=0)
            gray = np.expand_dims(gray, axis=-1)

            prediction = emotion_model.predict(gray, verbose=0)[0]
            label = emotion_labels[np.argmax(prediction)]

            face_buffer.append(label)
            emotion = max(set(face_buffer), key=face_buffer.count)

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            break  # Only process the first detected face

    return emotion
