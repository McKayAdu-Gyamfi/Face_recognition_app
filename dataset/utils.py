import os
import pickle as pkl
import cv2
import numpy as np
import yaml
from dataset.facenet_utils import load_facenet_model, detect_faces, get_face_embedding

class FaceDataHandler:
    def __init__(self, config_path='config.yaml'):
        # Load config
        self.cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.dataset_dir = self.cfg['PATH']['DATASET_DIR']
        self.pkl_path = self.cfg['PATH']['PKL_PATH']
        # Initialize FaceNet model
        self.model = load_facenet_model()
        self.dataset = {}  # Use dataset instead of information for clarity
        # Load existing dataset if available
        self.get_database()

    def get_database(self):
        if os.path.exists(self.pkl_path):
            with open(self.pkl_path, 'rb') as f:
                self.dataset = pkl.load(f)
        else:
            self.dataset = {}
        return self.dataset

    def save_database(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.pkl_path), exist_ok=True)
        with open(self.pkl_path, 'wb') as f:
            pkl.dump(self.dataset, f)

    def build_dataset(self):
        self.dataset.clear()
        counter = 0

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            print(f"Created dataset directory at {self.dataset_dir}")
            return  # No data yet

        # Loop through subdirectories (each subdirectory is one person)
        for person_folder in os.listdir(self.dataset_dir):
            person_path = os.path.join(self.dataset_dir, person_folder)
            if not os.path.isdir(person_path):
                continue

            person_id = person_folder
            person_name = person_folder

            for image_name in os.listdir(person_path):
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                image_path = os.path.join(person_path, image_name)

                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Failed to load image: {image_path}")
                        continue

                    faces = detect_faces(image)
                    if len(faces) == 0:
                        print(f"No face detected in: {image_path}")
                        continue

                    face_img = self.extract_face(image, faces[0])
                    embedding = get_face_embedding(image, faces[0], self.model)

                    self.dataset[counter] = {
                        'encoding': embedding,
                        'id': person_id,
                        'name': person_name,
                        'filename': image_name
                    }
                    print(f"Processed {image_name} for {person_name}")
                    counter += 1
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        self.save_database()
        print(f"Dataset built with {counter} faces")

    def extract_face(self, image, face_bbox):
        # face_bbox = [x, y, w, h]
        x, y, w, h = face_bbox
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))  # Resize for FaceNet
        return face_img

    def delete_dataset(self):
        if os.path.exists(self.pkl_path):
            os.remove(self.pkl_path)
        self.dataset.clear()

    def load_embeddings(self):
        # Return all embeddings as numpy array
        return np.array([v['encoding'] for v in self.dataset.values()])
