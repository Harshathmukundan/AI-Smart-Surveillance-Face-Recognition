import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Increase frame size for better detection
        self.frame_resizing = 0.5  # Changed from 0.25 to 0.5 for better quality

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("Looking for images in:", images_path)

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            print(f"Processing image: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
                
            # Increase image size for better face detection
            img = cv2.resize(img, (0, 0), fx=2, fy=2)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            print(f"Processing file: {filename}{ext}")
            
            # Get encoding
            try:
                faces = face_recognition.face_encodings(rgb_img, num_jitters=2)  # Added num_jitters for better accuracy
                if len(faces) == 0:
                    print(f"No faces found in {img_path}")
                    continue
                img_encoding = faces[0]
                
                # Store file name and file encoding
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
                print(f"Successfully encoded face from {filename}{ext}")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
                
        print(f"Encoding images loaded. Total faces encoded: {len(self.known_face_encodings)}")
        print(f"Known face names: {self.known_face_names}")

    def detect_known_faces(self, frame):
        # Increase frame size for better detection
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        
        # Convert the image from BGR color to RGB color
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=2)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)  # Increased tolerance
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # More lenient matching
            if matches[best_match_index] and face_distances[best_match_index] < 0.6:  # Increased threshold
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
