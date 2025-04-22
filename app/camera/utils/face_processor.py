import cv2
import time
import face_recognition
import numpy as np
from app.database.members import get_all_face_encodings
from app.database.members import get_member
from app.database.attendance import record_attendance
from flask import current_app
import os

class FaceProcessor:
    """Handles face detection, recognition, and tracking functionalities."""
    
    def __init__(self):
        # Face recognition settings
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.face_thumbnails = []
        
        # Known face encodings and names
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.known_face_images = {}  # Store member images from database: {member_id: image}
        
        # Face similarity threshold (lower = more strict matching)
        self.face_similarity_threshold = 0.6
        
        # Persistent face tracking
        self.persistent_faces = {
            "known": {},    # Format: {name: {"image": thumbnail, "in_view": bool, "last_seen": timestamp, "encoding": face_encoding}}
            "unknown": []   # Format: [{"id": unique_id, "image": thumbnail, "in_view": bool, "last_seen": timestamp, "encoding": face_encoding}]
        }
        self.unknown_face_counter = 0
        
        # Latest recognition result
        self.last_recognition_result = None
        
        # Load faces from database
        self.load_known_faces_from_db()
    
    def load_known_faces_from_db(self):
        """Load known face encodings, names and member images from the database."""
        try:
            # Get face encodings and member data
            encodings, names, member_ids = get_all_face_encodings()
            self.known_face_encodings = encodings
            self.known_face_names = names
            self.known_face_ids = member_ids
            
            # We're keeping persistent_faces["known"] empty until faces are actually seen by the camera
            # This ensures only faces seen during this session appear in the UI
            
            # Clear existing images dictionary
            self.known_face_images = {}
            
            # Load images for all members for display purposes
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
                except Exception as img_error:
                    print(f"Error loading image for member {member_id}: {img_error}")
                    
            return True
        except Exception as e:
            print(f"Error loading faces from database: {e}")
            return False
    
    def reset_state(self):
        """Reset all recognition state."""
        print("Resetting face recognition state...")
        # Clear persistent faces
        self.persistent_faces = {
            "known": {},
            "unknown": []
        }
        
        # Reset counters and state
        self.unknown_face_counter = 0
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.face_thumbnails = []
        
        # Reload faces from DB to make sure we have the latest
        self.load_known_faces_from_db()
        
        # Clear recognition result
        self.last_recognition_result = None
    
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
    
    def clean_old_faces(self, max_age_seconds=300):
        """Remove unknown faces that haven't been seen for a while and are not in view."""
        current_time = time.time()
        
        # Clean up old unknown faces
        self.persistent_faces["unknown"] = [
            face for face in self.persistent_faces["unknown"]
            if face["in_view"] or (current_time - face["last_seen"] < max_age_seconds)
        ]
    
    def process_frame(self, frame):
        """Process a frame for face recognition.
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            processed_frame: Frame with face boxes drawn
            recognized_ids: List of recognized member IDs
        """
        try:
            # Reset current frame data
            self.face_names = []
            self.face_thumbnails = []
            recognized_ids = []
            
            # Mark all persistent faces as not in view for this frame
            for name in self.persistent_faces["known"]:
                self.persistent_faces["known"][name]["in_view"] = False
                
            for unknown_face in self.persistent_faces["unknown"]:
                unknown_face["in_view"] = False
            
            # Resize frame for faster face recognition
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert from BGR (OpenCV) to RGB (face_recognition)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in the current frame
            self.face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
            
            # Create thumbnails of each face
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
                
                # Save the thumbnail
                if face_image.size > 0:
                    current_thumbnails.append(face_image)
                else:
                    current_thumbnails.append(None)  # Placeholder for invalid thumbnails
            
            # Compare against known faces
            if len(self.known_face_encodings) > 0:
                for face_encoding in self.face_encodings:
                    # Get valid encodings
                    valid_encodings, valid_names, valid_ids = self._get_valid_encodings()
                    
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
                    
                    self.face_names.append(name)
                    
                    # Update persistent face tracking
                    face_idx = len(self.face_names) - 1
                    thumbnail = current_thumbnails[face_idx] if face_idx < len(current_thumbnails) else None
                    
                    if thumbnail is not None:
                        self._update_persistent_faces(name, member_id, face_encoding, thumbnail)
            else:
                # If no known faces, mark all as unknown
                self.face_names = ["Unknown"] * len(self.face_locations)
                
                # Process unknown faces
                for i, thumbnail in enumerate(current_thumbnails):
                    if thumbnail is not None and i < len(self.face_encodings):
                        self._process_unknown_face(self.face_encodings[i], thumbnail)
            
            # Update recognition result timestamp
            current_time = time.time()
            
            # Create the processed frame with drawn boxes
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
            
            # Update recognition result with all faces data
            self._update_recognition_result(current_time)
            
            return processed_frame, recognized_ids
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Clear arrays on error
            self.face_locations = []
            self.face_encodings = []
            self.face_names = []
            self.face_thumbnails = []
            return frame.copy(), []
    
    def _get_valid_encodings(self):
        """Get valid face encodings, names, and IDs.
        
        Returns:
            valid_encodings: List of valid face encodings
            valid_names: List of corresponding names
            valid_ids: List of corresponding member IDs
        """
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
        
        return valid_encodings, valid_names, valid_ids
    
    def _update_persistent_faces(self, name, member_id, face_encoding, thumbnail):
        """Update persistent face tracking data.
        
        Args:
            name: Name of the person
            member_id: Member ID or None for unknown
            face_encoding: Face encoding
            thumbnail: Face thumbnail image
        """
        current_time = time.time()
        
        if name != "Unknown":
            # Update or create known face entry
            if name not in self.persistent_faces["known"]:
                self.persistent_faces["known"][name] = {
                    "image": thumbnail,
                    "in_view": True,
                    "last_seen": current_time,
                    "member_id": member_id,
                    "encoding": face_encoding
                }
            else:
                # Person already known, update tracking
                self.persistent_faces["known"][name]["in_view"] = True
                self.persistent_faces["known"][name]["last_seen"] = current_time
                self.persistent_faces["known"][name]["encoding"] = face_encoding
                
                # Only update thumbnail if it's significantly better quality
                if thumbnail.size > self.persistent_faces["known"][name]["image"].size * 1.2:
                    self.persistent_faces["known"][name]["image"] = thumbnail
        else:
            # Handle unknown face - check if it matches existing unknown faces
            self._process_unknown_face(face_encoding, thumbnail)
    
    def _process_unknown_face(self, face_encoding, thumbnail):
        """Process an unknown face.
        
        Args:
            face_encoding: Face encoding
            thumbnail: Face thumbnail image
        """
        current_time = time.time()
        
        # First check if it matches an existing unknown face
        matched_face = self._match_unknown_face(face_encoding)
        
        if matched_face:
            # Update existing face
            matched_face["in_view"] = True
            matched_face["last_seen"] = current_time
            matched_face["encoding"] = face_encoding
            # Only update if better quality
            if thumbnail.size > matched_face["image"].size * 1.2:
                matched_face["image"] = thumbnail
        else:
            # Create new unknown face
            self.unknown_face_counter += 1
            unknown_id = f"unknown_{self.unknown_face_counter}"
            
            # Ensure we have a valid face encoding
            if face_encoding is not None:
                new_unknown = {
                    "id": unknown_id,
                    "image": thumbnail,
                    "in_view": True,
                    "last_seen": current_time,
                    "encoding": face_encoding
                }
                print(f"Added new unknown face {unknown_id}")
                self.persistent_faces["unknown"].append(new_unknown)
            else:
                print(f"Warning: Tried to add unknown face {unknown_id} but encoding is None")
    
    def _update_recognition_result(self, current_time):
        """Update the recognition result with current face data.
        
        Args:
            current_time: Current timestamp
        """
        # Combine known and unknown persistent faces into a single result
        persistent_faces_data = []
        
        # Add known faces - only those who have been seen this session
        for name, face_data in self.persistent_faces["known"].items():
            # Only include faces that have been seen (last_seen > 0)
            if face_data["last_seen"] > 0:
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
                "thumbnail_id": face_data["id"]  # Use unique ID
            })
        
        # Create the full result 
        self.last_recognition_result = {
            "timestamp": current_time,
            "faces": persistent_faces_data
        }
    
    def get_face_thumbnail(self, thumbnail_id):
        """Return a specific face thumbnail as JPEG bytes.
        
        Args:
            thumbnail_id: Either a name (for known faces) or unique ID (for unknown faces)
            
        Returns:
            JPEG bytes of the thumbnail image
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
    
    def record_attendance(self, meeting_id=None):
        """Record attendance for all recognized members.
        
        Args:
            meeting_id: Optional meeting ID (default: current active meeting)
            
        Returns:
            Number of members for whom attendance was recorded
        """
        recorded_count = 0
        all_recognized_ids = set()
        
        # Collect all member IDs from persistent known faces
        for _, face_data in self.persistent_faces["known"].items():
            if face_data.get("member_id") and face_data.get("last_seen", 0) > 0:
                all_recognized_ids.add(face_data["member_id"])
        
        # Only try to record attendance if there are recognized members
        if all_recognized_ids:
            with current_app.app_context():
                for member_id in all_recognized_ids:
                    try:
                        success = record_attendance(member_id, meeting_id)
                        if success:
                            recorded_count += 1
                            print(f"Recorded attendance for member ID: {member_id}")
                    except Exception as e:
                        print(f"Error recording attendance: {e}")
        
        return recorded_count
