import streamlit as st
import cv2
import numpy as np
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model

def preprocess_image(image, target_size=(160, 160)):
    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size)
    
    # Convert the pixel values to float32
    resized_image = resized_image.astype('float32')
    
    # Normalize the pixel values to the range [0, 1]
    normalized_image = resized_image / 255.0
    
    return normalized_image

def extract_face_embeddings(video_path, face_net, max_faces=5):
    cap = cv2.VideoCapture(video_path)
    face_embeddings = []
    faces_detected = 0

    while faces_detected < max_faces:
        ret, frame = cap.read()
        if not ret:
            break
        
        preprocessed_frame = preprocess_image(frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        
        embedding = face_net.embeddings(preprocessed_frame)
        
        if embedding is not None:
            faces_detected += 1
            face_embeddings.append(embedding)
    
    cap.release()
    return np.array(face_embeddings)

# Load the FaceNet model
face_net = FaceNet()

# Load your pre-trained LSTM model
model_path = "lstm_model.h5"
model = load_model(model_path)

st.title("Deepfake Detection")

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file is not None:
    temp_video_path = "uploaded_video.mp4"
    
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())
    
    st.video(temp_video_path)
    
    if st.button("Detect Deepfake"):
        face_embeddings = extract_face_embeddings(temp_video_path, face_net)
        
        if len(face_embeddings) > 0:
            predictions = model.predict(face_embeddings)
            
            # Convert predictions to "Real" or "Fake"
            results = ["Real" if pred > 0.5 else "Fake" for pred in predictions]
            
            # Determine the overall result
            if results.count("Fake") > results.count("Real"):
                st.write("Overall Prediction: Fake")
            else:
                st.write("Overall Prediction: Real")
        else:
            st.write("No faces detected in the video.")