# app/voice_utils.py
import numpy as np
import librosa
from collections import deque

emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'calm']

# Optional: Use a buffer to smooth predictions over time
emotion_buffer = deque(maxlen=3)

def predict_voice_emotion(audio, model, smoothing=True):
    # Step 1: Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
    max_pad_len = 174
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    # Step 2: Predict
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model.predict(mfcc, verbose=0)[0]  # shape: (8,)
    top_index = np.argmax(prediction)
    label = emotion_labels[top_index]

    # Step 3: Smooth prediction if enabled
    if smoothing:
        emotion_buffer.append(label)
        most_common = max(set(emotion_buffer), key=emotion_buffer.count)
        return most_common
    else:
        return label