import streamlit as st
import cv2
import numpy as np
import yaml
import os
import shutil
import zipfile
import pandas as pd
from dataset.utils import FaceDataHandler

# Load model
from dataset.facenet_utils import detect_faces, get_face_embedding

st.set_page_config(layout="wide")
# Config
cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']
DATASET_DIR = cfg['PATH']['DATASET_DIR']

# Initialize dataset handler
handler = FaceDataHandler()

# Load FaceNet model
model = handler.model  # Reuse the model already loaded by handler

# Page title
st.title("Face Recognition App")

# Session states
if 'recognized' not in st.session_state:
    st.session_state['recognized'] = False
    st.session_state['authorized'] = False
if 'webcam_active' not in st.session_state:
    st.session_state['webcam_active'] = False

# Function to recognize faces
def recognize_face(image, dataset_embeddings, threshold=0.4):
    """
    Recognize faces in the image by comparing with dataset embeddings.
    Returns: (authorized, recognition_info)
    """
    faces = detect_faces(image)
    if len(faces) == 0:
        return False, {"message": "No faces detected"}
    
    authorized = False
    recognition_info = {"faces": len(faces), "recognized": []}
    
    # Load names and IDs from CSV for matching
    try:
        df = pd.read_csv(handler.embeddings_csv)
        ids = df['id'].tolist() if 'id' in df.columns else []
        names = df['name'].tolist() if 'name' in df.columns else []
    except Exception:
        # Fall back to handler.dataset if CSV fails
        ids = [v['id'] for v in handler.dataset.values()]
        names = [v['name'] for v in handler.dataset.values()]
    
    for i, face in enumerate(faces):
        # Get embedding for this face
        emb = get_face_embedding(image, face, model)
        
        # Compare with all stored embeddings
        if dataset_embeddings.size > 0:
            distances = np.linalg.norm(dataset_embeddings - emb, axis=1)
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            
            face_info = {
                "face_num": i+1,
                "confidence": 1.0 - min(min_dist, threshold) / threshold
            }
            
            if min_dist <= threshold:
                authorized = True
                face_info["recognized"] = True
                face_info["name"] = names[min_dist_idx] if min_dist_idx < len(names) else "Unknown"
                face_info["id"] = ids[min_dist_idx] if min_dist_idx < len(ids) else "Unknown"
                face_info["distance"] = float(min_dist)
            else:
                face_info["recognized"] = False
                face_info["distance"] = float(min_dist)
            
            recognition_info["recognized"].append(face_info)
    
    return authorized, recognition_info

# Sidebar
st.sidebar.title("Settings")
menu = ["Picture", "Webcam", "Manage Dataset"]
choice = st.sidebar.selectbox("Mode", menu)

# Authorization indicator
st.sidebar.title("Authorization Status:")
status_container = st.sidebar.empty()

# Get embeddings from dataset
dataset_embeddings = np.array([v['encoding'] for v in handler.dataset.values()]) if handler.dataset else np.array([])

# Main content based on selected mode
if choice == "Picture":
    st.write(PICTURE_PROMPT)
    uploaded_image = st.file_uploader("Upload an image", type=['jpg','png','jpeg'])
    
    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Display image
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if dataset_embeddings.size > 0:
            recognized = recognize_face(image, dataset_embeddings)
            st.session_state['recognized'] = True
            st.session_state['authorized'] = recognized
        else:
            st.warning("No dataset found, please build dataset using the 'Manage Dataset' option.")

elif choice == "Webcam":
    st.write(WEBCAM_PROMPT)
    
    # Create a button to start/stop webcam
    if st.button('Toggle Webcam'):
        st.session_state['webcam_active'] = not st.session_state['webcam_active']
    
    FRAME_WINDOW = st.empty()
    
    # Check if webcam should be active
    if st.session_state['webcam_active']:
        try:
            cam = cv2.VideoCapture(0)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # This is a single frame capture - Streamlit will rerun this 
            # on each interaction without getting stuck in a loop
            ret, frame = cam.read()
            
            if ret:
                # Process frame for recognition
                if dataset_embeddings.size > 0:
                    recognized = recognize_face(frame, dataset_embeddings)
                    st.session_state['recognized'] = True
                    st.session_state['authorized'] = recognized
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb)
            else:
                st.error("Failed to capture frame from webcam.")
                st.session_state['webcam_active'] = False
            
            # Release camera
            cam.release()
            
            # Add rerun to continuously update (with a slight delay to avoid excessive resource usage)
            if st.session_state['webcam_active']:
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Error accessing webcam: {e}")
            st.session_state['webcam_active'] = False
    else:
        # Show placeholder when webcam is not running
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, "Click 'Toggle Webcam' to start", (80, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        FRAME_WINDOW.image(placeholder_img)

elif choice == "Manage Dataset":
    st.subheader("Build Face Recognition Dataset")
    
    # Stats about current dataset
    if handler.dataset:
        st.success(f"Current dataset contains {len(handler.dataset)} face(s)")
    else:
        st.info("No dataset found. Upload images to build one.")
    
    # File uploader for multiple images
    st.subheader("Upload Folder for Dataset")    
    st.write("Upload images of faces to add to your dataset. Each image should be named in format: ID_Name.jpg")
    st.write("For example: '001_John_Smith.jpg'")
    
    uploaded_zip = st.file_uploader("Choose a ZIP file", type=['zip'])

    if uploaded_zip:
        st.write(f"Uploaded ZIP file: {uploaded_zip.name}")
        
        # Process button
        if st.button("Process ZIP and Build Dataset"):
            # Clean previous uploads
            if os.path.exists(handler.upload_dir):
                for item in os.listdir(handler.upload_dir):
                    item_path = os.path.join(handler.upload_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
            
            # Ensure upload directory exists
            os.makedirs(handler.upload_dir, exist_ok=True)
            
            # Save ZIP file to uploads directory
            zip_path = os.path.join(handler.upload_dir, uploaded_zip.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            
            st.write(f"ZIP file saved to: {zip_path}")
            
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                extract_dir = os.path.join(handler.upload_dir, "extracted")
                os.makedirs(extract_dir, exist_ok=True)
                zip_ref.extractall(extract_dir)
            
            st.write("ZIP file extracted successfully.")
            
            # Display extracted folder structure
            extracted_files = []
            for root, dirs, files in os.walk(extract_dir):
                level = root.replace(extract_dir, "").count(os.sep)
                indent = " " * 4 * level
                folder_name = os.path.basename(root)
                if folder_name:  # Skip the empty string for the root directory
                    st.write(f"{indent}{folder_name}/")
                sub_indent = " " * 4 * (level + 1)
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        extracted_files.append(os.path.join(root, file))
                        st.write(f"{sub_indent}{file}")
            
            # Build dataset
            with st.spinner(f"Processing {len(extracted_files)} images and building embeddings..."):
                faces_processed = handler.build_dataset(scan_uploads=True)
                # Refresh embeddings
                dataset_embeddings = handler.load_embeddings()
            
            # Show results
            st.success(f"Dataset built successfully with {faces_processed} face(s)")
            
            # Show output directory structure
            st.subheader("Output Directory Structure")
            st.write(f"1. Original ZIP: `{zip_path}`")
            st.write(f"2. Extracted Data: `{extract_dir}`")
            st.write(f"3. Processed Faces: `{handler.processed_dir}`")
            st.write(f"4. Embeddings CSV: `{handler.embeddings_csv}`")
            
            # Display CSV preview if it exists
            if os.path.exists(handler.embeddings_csv):
                try:
                    df = pd.read_csv(handler.embeddings_csv)
                    if 'embedding' in df.columns:
                        # Truncate embedding column for display
                        df_display = df.copy()
                        df_display['embedding'] = df_display['embedding'].str[:20] + '...'
                        st.write("CSV Preview:")
                        st.dataframe(df_display)
                except Exception as e:
                    st.error(f"Error displaying CSV: {e}")
    
    # Option to clear dataset
    st.subheader("Manage Existing Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("View Dataset Files"):
            try:
                # Show processed files
                processed_files = os.listdir(handler.processed_dir) if os.path.exists(handler.processed_dir) else []
                image_files = [f for f in processed_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if image_files:
                    st.write(f"Found {len(image_files)} processed image(s):")
                    for img_file in image_files:
                        st.write(f"- {img_file}")
                else:
                    st.info("No processed image files found.")
                    
                # Show CSV if it exists
                if os.path.exists(handler.embeddings_csv):
                    st.write(f"Embeddings CSV: {handler.embeddings_csv}")
                    try:
                        df = pd.read_csv(handler.embeddings_csv)
                        st.write(f"CSV contains {len(df)} entries")
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")
            except Exception as e:
                st.error(f"Error listing dataset files: {e}")
    
    with col2:
        if st.button("Clear Dataset"):
            if st.session_state.get('confirm_clear', False):
                try:
                    # Clean up using handler method
                    handler.delete_dataset()
                    dataset_embeddings = np.array([])
                    
                    st.success("Dataset cleared successfully")
                    st.session_state['confirm_clear'] = False
                except Exception as e:
                    st.error(f"Error clearing dataset: {e}")
            else:
                st.warning("Click again to confirm clearing the entire dataset")
                st.session_state['confirm_clear'] = True

# Show authorization status
if st.session_state['recognized']:
    if st.session_state['authorized']:
        status_container.success("Authorized")
    else:
        status_container.error("Unauthorized")
else:
    status_container.info("Waiting for recognition...")