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
        self.known_face_images = {}  # Store member images from database: {member_id: image}
        
        # Recognition results
        self.last_recognition_result = None
        
        # Store face thumbnails and tracking data
        self.face_thumbnails = []
        
        # Persistent face tracking
        self.persistent_faces = {
            "known": {},    # Format: {name: {"image": thumbnail, "in_view": bool, "last_seen": timestamp, "encoding": face_encoding}}
            "unknown": []   # Format: [{"id": unique_id, "image": thumbnail, "in_view": bool, "last_seen": timestamp, "encoding": face_encoding}]
        }
        self.unknown_face_counter = 0
        
        # Face similarity threshold for tracking (lower = more strict matching)
        self.face_similarity_threshold = 0.6
        
        # Load known faces from database
        self.load_known_faces_from_db()
    
    def load_known_faces_from_db(self):
        """Load known face encodings, names and member images from the database."""
        try:
            # Get face encodings and member data
            encodings, names, member_ids = get_all_face_encodings()
            self.known_face_encodings = encodings
            self.known_face_names = names
            self.known_face_ids = member_ids
            
            # Reset the known faces dictionary in persistent faces
            # (we'll rebuild it with the latest DB data)
            self.persistent_faces["known"] = {}
            
            # Load member images and create persistent known face entries
            from app.database.members import get_member
            import os
            import cv2
            from flask import current_app
            
            for i, member_id in enumerate(member_ids):
                if member_id is None:
                    continue
                
                try:
                    # Get member details including image path
                    member = get_member(member_id)
                    if member and member['image_path']:
                        # Construct the full image path
                        image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], member['image_path'])
                        
                        # Load the image if it exists
                        if os.path.exists(image_path):
                            image = cv2.imread(image_path)
                            if image is not None:
                                # Store the image for this member
                                self.known_face_images[member_id] = image
                                
                                # Add or update entry in persistent known faces
                                name = member['name']
                                self.persistent_faces["known"][name] = {
                                    "image": image,
                                    "in_view": False,
                                    "last_seen": 0,  # Will be updated when seen
                                    "member_id": member_id,
                                    "encoding": encodings[i] if i < len(encodings) else None
                                }
                except Exception as img_error:
                    print(f"Error loading image for member {member_id}: {img_error}")
                    
            return True
        except Exception as e:
            print(f"Error loading faces from database: {e}")
            return False
        
    def _match_unknown_face(self, face_encoding):
        """Match a face encoding against existing unknown faces.
        
        Args:
            face_encoding: The face encoding to match.
            
        Returns:
            The matched face entry if found, None otherwise.
        """
        # If we don't have any unknown faces yet, there's nothing to match against
        if not self.persistent_faces["unknown"]:
            return None
        
        # Get encodings from all unknown faces
        unknown_encodings = []
        for face in self.persistent_faces["unknown"]:
            if "encoding" in face and face["encoding"] is not None:
                unknown_encodings.append((face, face["encoding"]))
        
        # No valid encodings to compare against
        if not unknown_encodings:
            return None
        
        # Compare the new face against all existing unknown faces
        try:
            # Calculate all face distances
            min_distance = float('inf')
            best_match = None
            
            for face_entry, encoding in unknown_encodings:
                # Calculate face distance
                face_distance = face_recognition.face_distance([encoding], face_encoding)[0]
                
                # Check if this is the best match so far
                if face_distance < min_distance:
                    min_distance = face_distance
                    best_match = face_entry
            
            # Return the best match if it's below our threshold
            if min_distance < self.face_similarity_threshold:
                return best_match
            
            # No matches found below threshold
            return None
        except Exception as e:
            print(f"Error matching unknown face: {e}")
            return None
            
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
        
    def _clean_old_faces(self, max_age_seconds=300):
        """Remove unknown faces that haven't been seen for a while and are not in view."""
        current_time = time.time()
        
        # Clean up old unknown faces
        self.persistent_faces["unknown"] = [
            face for face in self.persistent_faces["unknown"]
            if face["in_view"] or (current_time - face["last_seen"] < max_age_seconds)
        ]
    
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
                    # Clean old faces periodically (every 100 frames or so)
                    if self.process_this_frame and (time.time() % 10 < 0.1):  # Roughly every 10 seconds
                        self._clean_old_faces()
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
                            
                            # Reset face names and thumbnails for current frame processing
                            self.face_names = []
                            self.face_thumbnails = []
                            recognized_ids = []
                            
                            # First, mark all persistent faces as not in view for this frame
                            for name in self.persistent_faces["known"]:
                                self.persistent_faces["known"][name]["in_view"] = False
                                
                            for unknown_face in self.persistent_faces["unknown"]:
                                unknown_face["in_view"] = False
                            
                            # Create thumbnails of each face for this frame
                            current_thumbnails = []
                            for (top, right, bottom, left) in self.face_locations:
                                # Scale back up face location since we resized by 1/4
                                scale = 4.0
                                top_scaled = int(top * scale)
                                right_scaled = int(right * scale)
                                bottom_scaled = int(bottom * scale)
                                left_scaled = int(left * scale)
                                
                                # Add some padding around the face
                                padding = 20
                                top_scaled = max(0, top_scaled - padding)
                                left_scaled = max(0, left_scaled - padding)
                                bottom_scaled = min(frame.shape[0], bottom_scaled + padding)
                                right_scaled = min(frame.shape[1], right_scaled + padding)
                                
                                # Extract the face image
                                face_image = frame[top_scaled:bottom_scaled, left_scaled:right_scaled]
                                
                                # Save the thumbnail for this frame's processing
                                if face_image.size > 0:
                                    current_thumbnails.append(face_image)
                                else:
                                    current_thumbnails.append(None)  # Placeholder for invalid thumbnails
                            
                            # Check if there are known faces to compare against
                            if len(self.known_face_encodings) > 0:
                                for face_encoding in self.face_encodings:
                                    try:
                                        # Ensure all encodings are numpy arrays with the same shape
                                        valid_encodings = []
                                        valid_names = []
                                        valid_ids = []
                                        
                                        for i, known_encoding in enumerate(self.known_face_encodings):
                                            try:
                                                # Convert any bytes to numpy arrays if needed
                                                if isinstance(known_encoding, bytes):
                                                    from app.database import convert_array
                                                    known_encoding = convert_array(known_encoding)
                                                
                                                # Verify it's a valid numpy array with the right shape
                                                if (isinstance(known_encoding, np.ndarray) and 
                                                    known_encoding.dtype == np.float64 and
                                                    len(known_encoding.shape) == 1):
                                                    valid_encodings.append(known_encoding)
                                                    valid_names.append(self.known_face_names[i])
                                                    valid_ids.append(self.known_face_ids[i] if i < len(self.known_face_ids) else None)
                                            except Exception as enc_error:
                                                print(f"Error with encoding {i}: {enc_error}")
                                                continue
                                        
                                        # Default to Unknown
                                        name = "Unknown"
                                        member_id = None
                                        
                                        # Only proceed if we have valid encodings
                                        if valid_encodings:
                                            matches = face_recognition.compare_faces(valid_encodings, face_encoding)
                                            face_distances = face_recognition.face_distance(valid_encodings, face_encoding)
                                            
                                            if len(face_distances) > 0:
                                                best_match_index = np.argmin(face_distances)
                                                if matches[best_match_index]:
                                                    name = valid_names[best_match_index]
                                                    # Record member ID for attendance
                                                    member_id = valid_ids[best_match_index]
                                                    if member_id is not None:
                                                        recognized_ids.append(member_id)
                                    except Exception as comp_error:
                                        print(f"Error comparing faces: {comp_error}")
                                        name = "Unknown"
                                        member_id = None
                                    
                                    self.face_names.append(name)
                                    
                                    # Update persistent known faces
                                    face_idx = len(self.face_names) - 1
                                    thumbnail = current_thumbnails[face_idx] if face_idx < len(current_thumbnails) else None
                                    
                                    if thumbnail is not None:
                                        # If this is a known person, update their persistent record
                                        current_time = time.time()
                                        if name != "Unknown":
                                            # Update or create the known face entry
                                            if name not in self.persistent_faces["known"]:
                                                self.persistent_faces["known"][name] = {
                                                    "image": thumbnail,
                                                    "in_view": True,
                                                    "last_seen": current_time,
                                                    "member_id": member_id
                                                }
                                            else:
                                                # Person already known, mark as in view and update last seen
                                                self.persistent_faces["known"][name]["in_view"] = True
                                                self.persistent_faces["known"][name]["last_seen"] = current_time
                                                # Only update thumbnail if it's significantly better quality
                                                # This keeps the image stable instead of constantly changing
                                                if thumbnail.size > self.persistent_faces["known"][name]["image"].size * 1.2:
                                                    self.persistent_faces["known"][name]["image"] = thumbnail
                                        else:
                                            # This is an unknown face - check if it matches any existing unknown faces
                                            # before creating a new entry
                                            matched_face = self._match_unknown_face(face_encoding)
                                            
                                            if matched_face:
                                                # Update the existing face entry
                                                matched_face["in_view"] = True
                                                matched_face["last_seen"] = current_time
                                                # Only update the thumbnail if it's significantly better
                                                if thumbnail.size > matched_face["image"].size * 1.2:
                                                    matched_face["image"] = thumbnail
                                            else:
                                                # No match found, create a new unknown face entry
                                                self.unknown_face_counter += 1
                                                unknown_id = f"unknown_{self.unknown_face_counter}"
                                                new_unknown = {
                                                    "id": unknown_id,
                                                    "image": thumbnail,
                                                    "in_view": True,
                                                    "last_seen": current_time,
                                                    "encoding": face_encoding
                                                }
                                                self.persistent_faces["unknown"].append(new_unknown)
                            else:
                                # If no known faces are loaded, just mark all faces as unknown
                                self.face_names = ["Unknown"] * len(self.face_locations)
                                
                                # Add all as unknown persistent faces
                                current_time = time.time()
                                for i, thumbnail in enumerate(current_thumbnails):
                                    if thumbnail is not None and i < len(self.face_encodings):
                                        # Try to match with existing unknown faces first
                                        face_encoding = self.face_encodings[i]
                                        matched_face = self._match_unknown_face(face_encoding)
                                        
                                        if matched_face:
                                            # Update the existing face entry
                                            matched_face["in_view"] = True
                                            matched_face["last_seen"] = current_time
                                            # Only update the thumbnail if it's significantly better
                                            if thumbnail.size > matched_face["image"].size * 1.2:
                                                matched_face["image"] = thumbnail
                                        else:
                                            # Create a new unknown face entry
                                            self.unknown_face_counter += 1
                                            unknown_id = f"unknown_{self.unknown_face_counter}"
                                            self.persistent_faces["unknown"].append({
                                                "id": unknown_id,
                                                "image": thumbnail,
                                                "in_view": True,
                                                "last_seen": current_time,
                                                "encoding": face_encoding
                                            })
                            
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
                            self.face_thumbnails = []
                    
                    # Store the recognition result
                    current_time = time.time()
                    
                    # Combine both known and unknown persistent faces into a single result
                    persistent_faces_data = []
                    
                    # Add known faces
                    for name, face_data in self.persistent_faces["known"].items():
                        persistent_faces_data.append({
                            "name": name,
                            "member_id": face_data["member_id"],
                            "in_view": face_data["in_view"],
                            "last_seen": face_data["last_seen"],
                            "time_since_seen": current_time - face_data["last_seen"],
                            "type": "known",
                            "thumbnail_id": name.replace(" ", "_").lower()  # Use name as ID for thumbnail retrieval
                        })
                    
                    # Add unknown faces
                    for face_data in self.persistent_faces["unknown"]:
                        persistent_faces_data.append({
                            "name": "Unknown",
                            "member_id": None,
                            "in_view": face_data["in_view"],
                            "last_seen": face_data["last_seen"],
                            "time_since_seen": current_time - face_data["last_seen"],
                            "type": "unknown",
                            "thumbnail_id": face_data["id"]  # Use unique ID for thumbnail retrieval
                        })
                    
                    # Create the full result 
                    self.last_recognition_result = {
                        "timestamp": current_time,
                        "faces": persistent_faces_data
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
        
    def get_face_thumbnail(self, thumbnail_id):
        """Return a specific face thumbnail as JPEG bytes.
        
        Args:
            thumbnail_id: Either a name (for known faces) or unique ID (for unknown faces)
        """
        try:
            # First check if it's a known face
            for name, data in self.persistent_faces["known"].items():
                name_id = name.replace(" ", "_").lower()
                if name_id == thumbnail_id or name == thumbnail_id:
                    # For known faces, use the database image if available
                    member_id = data.get("member_id")
                    if member_id and member_id in self.known_face_images:
                        # Use member image from database
                        ret, jpeg = cv2.imencode('.jpg', self.known_face_images[member_id])
                        if ret:
                            return jpeg.tobytes()
                    
                    # Fallback to the persistent thumbnail
                    ret, jpeg = cv2.imencode('.jpg', data["image"])
                    if ret:
                        return jpeg.tobytes()
            
            # Then check unknown faces
            for face in self.persistent_faces["unknown"]:
                if face["id"] == thumbnail_id:
                    # Return the unknown face thumbnail
                    ret, jpeg = cv2.imencode('.jpg', face["image"])
                    if ret:
                        return jpeg.tobytes()
                        
            # Fallback for numeric indices (for backward compatibility)
            try:
                index = int(thumbnail_id)
                if 0 <= index < len(self.face_thumbnails):
                    ret, jpeg = cv2.imencode('.jpg', self.face_thumbnails[index])
                    if ret:
                        return jpeg.tobytes()
            except (ValueError, TypeError):
                pass
                
        except Exception as e:
            print(f"Error getting face thumbnail '{thumbnail_id}': {e}")
            
        return None
    
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