import os
import pickle as pkl
import cv2
import numpy as np
import yaml
import pandas as pd
import shutil
from dataset.facenet_utils import load_facenet_model, detect_faces, get_face_embedding

class FaceDataHandler:
    def __init__(self, config_path='config.yaml'):
        # Load config
        self.cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.dataset_dir = self.cfg['PATH']['DATASET_DIR']
        self.processed_dir = self.cfg['PATH']['PROCESSED_DIR']
        self.upload_dir = self.cfg['PATH']['UPLOAD_DIR']
        self.embeddings_csv = self.cfg['PATH']['EMBEDDINGS_CSV']
        self.pkl_path = self.cfg['PATH']['PKL_PATH']
        
        # Initialize FaceNet model
        self.model = load_facenet_model()
        self.dataset = {}  # Use dataset instead of information for clarity
        
        # Load existing dataset if available
        self.get_database()
        
        # Ensure directories exist
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)

    def get_database(self):
        """Load the existing dataset from pickle file if it exists"""
        if os.path.exists(self.pkl_path):
            try:
                with open(self.pkl_path, 'rb') as f:
                    self.dataset = pkl.load(f)
                print(f"Loaded dataset with {len(self.dataset)} entries")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                self.dataset = {}
        else:
            print("No existing dataset found")
            self.dataset = {}
        return self.dataset

    def save_database(self):
        """Save the current dataset to a pickle file and CSV"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.pkl_path), exist_ok=True)
        
        try:
            # Save pickle file
            with open(self.pkl_path, 'wb') as f:
                pkl.dump(self.dataset, f)
            
            # Save as CSV
            if self.dataset:
                # Create a list to store the data
                csv_data = []
                for idx, data in self.dataset.items():
                    # Flatten embedding to string format
                    embedding_str = ','.join(map(str, data['encoding'].flatten()))
                    
                    # Add row for this person
                    csv_data.append({
                        'id': data['id'],
                        'name': data['name'],
                        'filename': data['filename'],
                        'embedding': embedding_str
                    })
                
                # Convert to DataFrame and save
                df = pd.DataFrame(csv_data)
                df.to_csv(self.embeddings_csv, index=False)
                
                print(f"Dataset saved with {len(self.dataset)} entries to CSV and PKL")
            else:
                # Create empty CSV if dataset is empty
                pd.DataFrame(columns=['id', 'name', 'filename', 'embedding']).to_csv(self.embeddings_csv, index=False)
                print("Empty dataset saved")
                
        except Exception as e:
            print(f"Error saving dataset: {e}")

    def process_image(self, image_path, dest_dir=None):
        """Process a single image and return face data"""
        image_name = os.path.basename(image_path)
        
        # Parse name/id from filename
        filename_no_ext = os.path.splitext(image_name)[0]
        parsed_name = filename_no_ext.split('_')
        
        # Extract ID and name
        if len(parsed_name) >= 2:
            person_id = parsed_name[0]
            person_name = ' '.join(parsed_name[1:])
        else:
            # If filename doesn't follow the expected format
            person_id = filename_no_ext
            person_name = filename_no_ext
        
        # Load and process image
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None
            
            # Detect faces
            faces = detect_faces(image)
            if len(faces) == 0:
                print(f"No face detected in: {image_name}")
                return None
            
            # Use the first face found
            face = faces[0]
            x, y, w, h = face
            face_img = image[y:y+h, x:x+w]
            
            # Create processed image - draw rectangle around face
            processed_img = image.copy()
            cv2.rectangle(processed_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(processed_img, person_name, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Save processed image if destination directory provided
            if dest_dir:
                processed_path = os.path.join(dest_dir, f"processed_{image_name}")
                cv2.imwrite(processed_path, processed_img)
                
                # Also save just the face
                face_path = os.path.join(dest_dir, f"face_{image_name}")
                cv2.imwrite(face_path, face_img)
            
            # Get embedding
            embedding = get_face_embedding(image, face, self.model)
            
            # Return the face data
            return {
                'encoding': embedding,
                'id': person_id,
                'name': person_name,
                'filename': image_name
            }
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            return None

    def build_dataset(self, scan_uploads=True):
        """Process all images and build face embeddings"""
        # Clear existing dataset
        self.dataset.clear()
        counter = 0
        
        # Clear processed directory
        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Define source directory - either upload dir or dataset dir
        source_dir = self.upload_dir if scan_uploads else self.dataset_dir
        
        # First, collect all valid image paths (including subdirectories)
        image_paths = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process each image
        for image_path in image_paths:
            face_data = self.process_image(image_path, self.processed_dir)
            if face_data:
                self.dataset[counter] = face_data
                counter += 1
                print(f"Processed: {face_data['filename']} - {face_data['name']} (ID: {face_data['id']})")
            
        # Save the updated dataset
        self.save_database()
        print(f"Dataset built with {counter} faces")
        return counter

    def delete_dataset(self):
        """Delete the dataset files and clear the dataset in memory"""
        files_to_delete = [self.pkl_path, self.embeddings_csv]
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        
        # Clear processed directory
        if os.path.exists(self.processed_dir):
            try:
                shutil.rmtree(self.processed_dir)
                os.makedirs(self.processed_dir, exist_ok=True)
                print(f"Cleared processed directory: {self.processed_dir}")
            except Exception as e:
                print(f"Error clearing processed directory: {e}")
        
        self.dataset.clear()
        print("Dataset cleared from memory")

    def load_embeddings(self):
        """Return all embeddings as numpy array"""
        if not self.dataset:
            return np.array([])
        return np.array([v['encoding'] for v in self.dataset.values()])
    
    def load_embeddings_from_csv(self):
        """Load embeddings from CSV file"""
        if not os.path.exists(self.embeddings_csv):
            print(f"CSV file not found: {self.embeddings_csv}")
            return np.array([])
            
        try:
            df = pd.read_csv(self.embeddings_csv)
            if 'embedding' not in df.columns or len(df) == 0:
                return np.array([])
                
            # Convert string embeddings back to numpy arrays
            embeddings = []
            for emb_str in df['embedding']:
                emb_values = list(map(float, emb_str.split(',')))
                embeddings.append(np.array(emb_values))
                
            return np.array(embeddings)
            
        except Exception as e:
            print(f"Error loading embeddings from CSV: {e}")
            return np.array([])