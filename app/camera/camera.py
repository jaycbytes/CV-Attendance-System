import cv2
import threading
import time
import face_recognition
import numpy as np
from app.database.members import get_all_face_encodings
from app.database.attendance import record_attendance

class Camera:
    """Base camera class for accessing webcam or USB cameras with face recognition."""
    
    def __init__(self, camera_id=0, recognition_enabled=False):
        self.camera_id = camera_id
        self.camera = None
        self.thread = None
        self.frame = None
        self.processed_frame = None  # Frame with face boxes drawn
        self.stopped = False
        
        # Face recognition settings
        self.recognition_enabled = recognition_enabled
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        
        # Known face encodings and names (to be populated)
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        # Recognition results
        self.last_recognition_result = None
        
        # Load known faces from database
        self.load_known_faces_from_db()
    
    def load_known_faces_from_db(self):
        """Load known face encodings and names from the database."""
        try:
            encodings, names, member_ids = get_all_face_encodings()
            self.known_face_encodings = encodings
            self.known_face_names = names
            self.known_face_ids = member_ids
            return True
        except Exception as e:
            print(f"Error loading faces from database: {e}")
            return False
        
    def load_known_faces(self, face_encodings, face_names):
        """Load known face encodings and names manually."""
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        
    def start(self):
        """Start the camera and capture thread."""
        self.camera = cv2.VideoCapture(self.camera_id)
        # Try to set higher resolution if supported
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera with ID {self.camera_id}")
        
        # Start the thread to read frames
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        return self
        
    def _capture_loop(self):
        """Capture frames in a loop and process for face recognition if enabled."""
        while not self.stopped:
            try:
                success, frame = self.camera.read()
                if not success:
                    # Add a small sleep before retry
                    time.sleep(0.1)
                    continue
                    
                # Store the original frame (make a copy to avoid threading issues)
                self.frame = frame.copy()
                
                # Process for face recognition if enabled
                if self.recognition_enabled:
                    # Only process every other frame to save processing power
                    if self.process_this_frame:
                        try:
                            # Resize frame for faster face recognition and to reduce memory usage
                            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                            
                            # Convert from BGR (OpenCV) to RGB (face_recognition)
                            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                            
                            # Find faces in the current frame
                            self.face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # Use faster HOG model
                            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                            
                            # Reset face names for this frame
                            self.face_names = []
                            recognized_ids = []
                            
                            # Check if there are known faces to compare against
                            if len(self.known_face_encodings) > 0:
                                for face_encoding in self.face_encodings:
                                    # Compare face with known faces
                                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                                    name = "Unknown"
                                    member_id = None
                                    
                                    # Use the known face with the smallest distance to the new face
                                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                                    if len(face_distances) > 0:
                                        best_match_index = np.argmin(face_distances)
                                        if matches[best_match_index]:
                                            name = self.known_face_names[best_match_index]
                                            # Record member ID for attendance
                                            if len(self.known_face_ids) > best_match_index:
                                                member_id = self.known_face_ids[best_match_index]
                                                if member_id is not None:  # Only record non-None IDs
                                                    recognized_ids.append(member_id)
                                    
                                    self.face_names.append(name)
                            else:
                                # If no known faces are loaded, just mark all faces as unknown
                                self.face_names = ["Unknown"] * len(self.face_locations)
                            
                            # Record attendance for recognized members
                            for member_id in recognized_ids:
                                try:
                                    record_attendance(member_id)
                                except Exception as e:
                                    print(f"Error recording attendance: {e}")
                        except Exception as e:
                            print(f"Error in face recognition processing: {e}")
                            # Set empty results in case of error to avoid crashing
                            self.face_locations = []
                            self.face_encodings = []
                            self.face_names = []
                    
                    # Store the recognition result
                    if len(self.face_names) > 0:
                        faces_data = []
                        for name, location in zip(self.face_names, self.face_locations):
                            member_id = None
                            # Safely look up member ID
                            if name != "Unknown" and name in self.known_face_names:
                                try:
                                    member_id = self.known_face_ids[self.known_face_names.index(name)]
                                except (ValueError, IndexError):
                                    print(f"Warning: Could not find member ID for {name}")
                                    
                            faces_data.append({
                                "name": name,
                                "location": location,
                                "member_id": member_id
                            })
                            
                        self.last_recognition_result = {
                            "timestamp": time.time(),
                            "faces": faces_data
                        }
                    else:
                        self.last_recognition_result = {
                            "timestamp": time.time(),
                            "faces": []
                        }
                
                    # Create a copy of the frame with boxes and labels drawn
                    processed_frame = frame.copy()
                    
                    # Draw the results on the processed frame
                    for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                        # Scale back up face locations since the frame we detected in was 1/4 size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        # Draw a box around the face
                        cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        
                        # Draw a label with a name below the face
                        cv2.rectangle(processed_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(processed_frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    
                    # Store the processed frame
                    self.processed_frame = processed_frame
                    
                    # Toggle flag for processing frames
                    self.process_this_frame = not self.process_this_frame
            except Exception as e:
                print(f"Error in camera capture loop: {e}")
                time.sleep(0.1)  # Pause briefly before continuing
            
    def get_frame(self, show_faces=False):
        """Convert frame to JPEG for MJPEG streaming.
        If show_faces is True and face recognition is enabled, return the processed frame
        with face boxes and labels drawn.
        """
        if show_faces and self.recognition_enabled and self.processed_frame is not None:
            frame_to_encode = self.processed_frame
        elif self.frame is not None:
            frame_to_encode = self.frame
        else:
            return None
            
        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame_to_encode)
        if not ret:
            return None
            
        return jpeg.tobytes()
    
    def get_recognition_result(self):
        """Return the last face recognition result."""
        return self.last_recognition_result
    
    def toggle_recognition(self, enabled=None):
        """Toggle face recognition processing."""
        if enabled is not None:
            self.recognition_enabled = enabled
        else:
            self.recognition_enabled = not self.recognition_enabled
            
        # If enabling recognition, reload faces from the database
        if self.recognition_enabled:
            self.load_known_faces_from_db()
            
        return self.recognition_enabled
    
    def get_camera_properties(self):
        """Return properties of the camera."""
        if not self.camera or not self.camera.isOpened():
            return {}
        
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        return {
            "id": self.camera_id,
            "width": width,
            "height": height,
            "fps": fps,
            "recognition_enabled": self.recognition_enabled
        }
        
    def stop(self):
        """Stop the camera thread and release resources."""
        self.stopped = True
        if self.thread:
            self.thread.join()
        if self.camera:
            self.camera.release()


def list_available_cameras(max_cameras=10):
    """List all available cameras by attempting to open each one."""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras