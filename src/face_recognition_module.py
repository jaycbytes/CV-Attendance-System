import face_recognition
import os
import numpy as np
import json
import cv2
import logging

# Configure logging
logging.basicConfig(
    filename='face_recognition.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def assess_face_quality(face_image):
    """Enhanced face quality assessment."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate core metrics
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()  # Sharpness
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Additional checks
        height, width = face_image.shape[:2]
        aspect_ratio = width / height
        min_dimension = min(width, height)
        
        # Penalize if:
        # - Image is too small
        # - Aspect ratio is too extreme
        # - Image is too blurry, dark, or low contrast
        quality_score = 0
        if min_dimension >= 60 and 0.7 <= aspect_ratio <= 1.3:
            quality_score = (
                (variance / 100) * 0.5 +  # Sharpness weight
                (brightness / 255) * 0.25 +  # Brightness weight
                (contrast / 128) * 0.25  # Contrast weight
            ) * 100
            
            # Additional penalties
            if variance < 100:  # Too blurry
                quality_score *= 0.5
            if brightness < 40 or brightness > 215:  # Too dark or too bright
                quality_score *= 0.7
            if contrast < 20:  # Too low contrast
                quality_score *= 0.7
                
        return quality_score
    except Exception as e:
        print(f"Error assessing face quality: {e}")
        return 0

def compare_face_similarity(encoding1, encoding2, tolerance=0.45):  # Stricter tolerance
    """Enhanced face similarity comparison."""
    try:
        if isinstance(encoding1, list):
            encoding1 = np.array(encoding1)
        if isinstance(encoding2, list):
            encoding2 = np.array(encoding2)
        
        # Calculate L2 distance
        distance = np.linalg.norm(encoding1 - encoding2)
        
        # Calculate cosine similarity
        similarity = 1 - distance
        
        # Use stricter threshold and additional checks
        if similarity >= (1 - tolerance):
            # Additional verification using different distance metric
            cosine_similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
            return cosine_similarity >= 0.85  # Additional threshold
            
        return False
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return False


def initialize_face_encodings(faces_dir):
    """Initialize face encodings from images in the faces directory."""
    print("Initializing face encodings...")
    encodings_file = os.path.join(faces_dir, "face_encodings.json")
    known_faces = {}

    # Process each image in the directory
    for filename in os.listdir(faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(faces_dir, filename)
            print(f"Processing image: {image_path}")

            try:
                # Load and encode face
                image = face_recognition.load_image_file(image_path)
                
                # Convert to RGB if needed
                if len(image.shape) == 2:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
                # Ensure we have a contiguous array
                if not image.flags['C_CONTIGUOUS']:
                    image = np.ascontiguousarray(image)

                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_faces[name] = {
                        "encoding": encodings[0].tolist(),
                        "major": "Computer Science"  # Default major
                    }
                    print(f"Successfully encoded face for {name}")
                else:
                    print(f"No face found in {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                import traceback
                print(traceback.format_exc())

    # Save encodings to JSON file
    if known_faces:
        try:
            with open(encodings_file, 'w') as f:
                json.dump(known_faces, f, indent=4)
            print(f"Saved encodings to {encodings_file}")
        except Exception as e:
            print(f"Error saving encodings: {e}")

    return known_faces

def load_known_faces(faces_dir):
    """Load known faces from the faces directory and JSON file."""
    encodings_file = os.path.join(faces_dir, "face_encodings.json")
    
    # If encodings file doesn't exist or is empty, initialize it
    if not os.path.exists(encodings_file) or os.path.getsize(encodings_file) == 0:
        return initialize_face_encodings(faces_dir)

    try:
        with open(encodings_file, 'r') as f:
            data = json.load(f)
            return {name: {
                'encoding': np.array(face_data['encoding']),
                'major': face_data['major']
            } for name, face_data in data.items()}
    except Exception as e:
        print(f"Error loading encodings, reinitializing: {e}")
        return initialize_face_encodings(faces_dir)