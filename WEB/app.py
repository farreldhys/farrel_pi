import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Pahlawan Indonesia Detection", page_icon="🦸‍♂️", layout="wide")

st.title("🦸‍♂️ Pahlawan Indonesia Detection")
st.write("Upload an image to detect Indonesian national heroes using the trained YOLOv9 model.")

@st.cache_resource
def load_model():
    # Load the model relative to the current directory
    return YOLO("best.pt")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Convert uploaded image to PIL Image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with st.spinner("Detecting..."):
        # Run inference
        results = model(image)
        
    with col2:
        st.subheader("Detection Results")
        for r in results:
            # Plot results
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            st.image(im, use_column_width=True)
            
            # Display detection summary
            boxes = r.boxes
            if len(boxes) > 0:
                st.write(f"**Detected {len(boxes)} object(s):**")
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    conf = float(box.conf[0])
                    st.write(f"- {class_name} (Confidence: {conf:.2f})")
            else:
                st.write("No objects detected.")
