import face_recognition
import os
import numpy as np
from cv2 import cvtColor, COLOR_BGR2RGB


# Load known faces of members from the images folder in order to encode and get names of each face
def load_known_faces(face_images_dir):
    known_face_encodings = []
    known_face_names = []
    
    # Loop through all images in the provided face_images_dir, 
    for image_name in os.listdir(face_images_dir):
        image_path = os.path.join(face_images_dir, image_name)
        # The reason we have to use load_image_file is to convert our images into numpy arrays which face_recognition library requires
        image = face_recognition.load_image_file(image_path) # images can be .png or .jpg, other ext haven't been tested
        image_encodings = face_recognition.face_encodings(image)
        
        if image_encodings:
            # Only the first encoding is appended because we're assuming there's only one face in the image from known_faces
            known_face_encodings.append(image_encodings[0])
            # The file name (without extension) is the person's name
            known_face_names.append(os.path.splitext(image_name)[0])
        else:
            print(f"Error: No face found in {image_name}")
    
    return known_face_encodings, known_face_names


# Function to process video frames and recognize faces
def frame_processing(video_capture, known_face_encodings, known_face_names, frame_queue, process_every_n_frames=3):
    frame_count = 0
    
    while True:
        ret, frame = video_capture.read()  # Capture frame from the camera
        # OpenCV video_capture returns a boolean (True or False) value for us to confirm if videos actually being captured
        if not ret: 
            break

        # Process only every nth frame to reduce load
        frame_count += 1
        if frame_count % process_every_n_frames == 0: # if the remainder isn't 0 then it's not a frame we want to process
            # Convert the frame to RGB
            rgb_frame = cvtColor(frame, COLOR_BGR2RGB)

            # Find all the face locations and encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            name = None

            # Loop over each face found in the frame to see if it's someone we know
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Put the frame and the recognized name (if any) into the queue
            frame_queue.put((frame, name))
        else:
            # If not processing for recognition, still put the frame in the queue
            frame_queue.put((frame, None))