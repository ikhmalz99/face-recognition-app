import streamlit as st
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

def augment_image(image_np):
    return np.fliplr(image_np)

def draw_box_with_name_simple(image_np, label, face_area):
    img = image_np.copy()
    x = face_area['x']
    y = face_area['y']
    w = face_area['w']
    h = face_area['h']
    
    label = os.path.splitext(label)[0][:15]
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(img, (x, y - text_height - 6), (x + text_width, y), (0, 255, 0), -1)
    
    cv2.putText(img, label, (x, y - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img

st.set_page_config(page_title="LFW Face Verification", layout="centered")
st.title("Face Recognition using VGG-Face + Augmentation (Lightweight Version)")
st.write("This app is optimized using the 'OpenCV' backend to prevent RAM crash issues on cloud servers.")

img1_file = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
img2_file = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

if img1_file and img2_file:
    img1 = Image.open(img1_file).convert("RGB")
    img2 = Image.open(img2_file).convert("RGB")

    st.image([img1, img2], caption=["Image 1 (Original)", "Image 2 (Original)"], width=250)

    img1_np = np.array(img1)
    img2_np = np.array(img2)

    img1_aug = augment_image(img1_np)
    img2_aug = augment_image(img2_np)

    with st.spinner("DeepFace is analyzing & verifying faces..."):
        try:
            res1 = DeepFace.verify(
                img1_np, img2_np, 
                model_name="VGG-Face", 
                detector_backend="opencv", 
                enforce_detection=True
            )
            
            res2 = DeepFace.verify(
                img1_aug, img2_aug, 
                model_name="VGG-Face", 
                detector_backend="opencv", 
                enforce_detection=True
            )
            
            avg_distance = (res1["distance"] + res2["distance"]) / 2
            match = avg_distance < 0.3

            st.success("Verification Process Completed!")
            
            col1, col2 = st.columns(2)
            with col1:
                if match:
                    st.metric(label="Match Result", value="MATCH (SAME)", delta="Identical Face")
                else:
                    st.metric(label="Match Result", value="NO MATCH", delta="- Different Person", delta_color="inverse")
            
            with col2:
                st.write(f"**Average Distance:** `{avg_distance:.4f}`")
                st.write(f"Original Distance: `{res1['distance']:.4f}`")
                st.write(f"Augmented Distance: `{res2['distance']:.4f}`")

            face_area1 = res1["facial_areas"]["img1"]
            face_area2 = res1["facial_areas"]["img2"]

            boxed1 = draw_box_with_name_simple(img1_np, os.path.basename(img1_file.name), face_area1)
            boxed2 = draw_box_with_name_simple(img2_np, os.path.basename(img2_file.name), face_area2)

            st.subheader("Face Detection Results")
            st.image([boxed1, boxed2], caption=["Detected Face 1", "Detected Face 2"], width=300)
            
        except ValueError as ve:
            st.error("Face not clearly detected in one of the images. Please ensure the face is looking straight at the camera and there is sufficient lighting.")
        except Exception as e:
            st.error(f"The system encountered an unexpected error: {e}")
