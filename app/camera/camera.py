import cv2
import threading
import time
import numpy as np
from app.camera.utils.camera_utils import try_camera_resolutions, set_camera_mjpeg, create_blank_frame
from app.camera.utils.face_processor import FaceProcessor

class Camera:
    """Base camera class for accessing webcam or USB cameras with face recognition."""
    
    def __init__(self, camera_id=0, recognition_enabled=False):
        self.camera_id = camera_id
        self.camera = None
        self.thread = None
        self.frame = None
        self.processed_frame = None  # Frame with face boxes drawn
        self.stopped = False
        
        # Camera error tracking
        self.frame_error_count = 0
        self.max_frame_errors = 10
        self.last_error = None
        
        # Face recognition
        self.recognition_enabled = recognition_enabled
        self.process_this_frame = True  # Process every other frame to save CPU
        self.face_processor = FaceProcessor()
        
        # Start the camera
        self.initialize_camera()
    
    def initialize_camera(self):
        """Initialize the camera with the best settings for the device."""
        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(self.camera_id)
            
            # Check if camera opened successfully
            if not self.camera.isOpened():
                raise RuntimeError(f"Could not open camera with ID {self.camera_id}")
            
            # Try to find best resolution
            print(f"Starting camera {self.camera_id} with optimal settings")
            try_camera_resolutions(self.camera)
            
            # Try to set MJPEG format for better compatibility
            set_camera_mjpeg(self.camera)
            
            # Create initial blank frame
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame = create_blank_frame(width, height, "Initializing camera...")
            
            print(f"Camera initialized with resolution: {width}x{height}")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.last_error = str(e)
            return False
    
    def start(self):
        """Start the camera capture thread."""
        # Make sure camera is initialized
        if self.camera is None or not self.camera.isOpened():
            self.initialize_camera()
        
        # Start the capture thread
        self.stopped = False
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        return self
    
    def _capture_loop(self):
        """Main loop for capturing frames from the camera."""
        while not self.stopped:
            try:
                # Read a frame from the camera
                success, frame = self.camera.read()
                
                # Handle frame read failures
                if not success:
                    self._handle_frame_error()
                    continue
                
                # Reset error counter on successful frame
                self.frame_error_count = 0
                
                # Validate frame
                if not self._validate_frame(frame):
                    continue
                
                # Store the original frame (make a copy to avoid threading issues)
                self.frame = frame.copy()
                
                # Process for face recognition if enabled
                if self.recognition_enabled:
                    if self.process_this_frame:
                        try:
                            # Clean old faces periodically
                            if time.time() % 10 < 0.1:  # Roughly every 10 seconds
                                self.face_processor.clean_old_faces()
                            
                            # Process the frame for face recognition
                            self.processed_frame, recognized_ids = self.face_processor.process_frame(frame)
                            
                        except Exception as e:
                            print(f"Error in face recognition processing: {e}")
                            self.last_error = str(e)
                            # Keep the original frame without processing
                            self.processed_frame = frame.copy()
                    
                    # Toggle flag to process every other frame (saves CPU)
                    self.process_this_frame = not self.process_this_frame
                
            except Exception as e:
                self.last_error = str(e)
                print(f"Error in camera capture loop: {e}")
                time.sleep(0.1)  # Pause briefly before continuing
    
    def _handle_frame_error(self):
        """Handle errors when reading frames."""
        # Increment error counter
        self.frame_error_count += 1
        
        # Log the error 
        print(f"Error reading frame from camera {self.camera_id} (error count: {self.frame_error_count})")
        
        # If we've had too many consecutive errors, try to reset the camera
        if self.frame_error_count >= self.max_frame_errors:
            print(f"Too many frame errors, attempting to reset camera {self.camera_id}...")
            # Release and reopen the camera
            if self.camera:
                self.camera.release()
            
            # Create a blank white frame with message
            self.frame = create_blank_frame(640, 480, "Camera reconnecting...")
            
            # Try to reopen the camera
            time.sleep(2.0)  # Give more time before reopening
            try:
                self.camera = cv2.VideoCapture(self.camera_id)
                print(f"Camera {self.camera_id} reset attempt")
                
                # Try setting to MJPG format which often works better
                set_camera_mjpeg(self.camera)
                
                self.frame_error_count = 0  # Reset error counter
            except Exception as reset_error:
                print(f"Error resetting camera: {reset_error}")
        
        # Small sleep before next attempt
        time.sleep(0.1)
    
    def _validate_frame(self, frame):
        """Validate a frame to ensure it's usable.
        
        Args:
            frame: OpenCV frame to validate
            
        Returns:
            bool: True if frame is valid, False otherwise
        """
        # Check if frame is None or empty
        if frame is None or frame.size == 0:
            print("Warning: Empty frame received, skipping")
            return False
        
        # Check for unusually small frames
        height, width = frame.shape[:2]
        if width < 10 or height < 10:
            print(f"Warning: Unusually small frame received ({width}x{height}), skipping")
            return False
        
        return True
    
    def get_frame(self, show_faces=False):
        """Convert frame to JPEG for MJPEG streaming.
        
        Args:
            show_faces: If True and recognition is enabled, show processed frame with face boxes
            
        Returns:
            JPEG bytes of the frame
        """
        # Choose which frame to encode
        if show_faces and self.recognition_enabled and self.processed_frame is not None:
            frame_to_encode = self.processed_frame
        elif self.frame is not None:
            frame_to_encode = self.frame
        else:
            # Create a blank frame with message if no frame is available
            frame_to_encode = create_blank_frame(640, 480, "Camera initializing...")
            
        try:    
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame_to_encode)
            if not ret:
                # Create a fallback frame if encoding fails
                blank_frame = create_blank_frame(640, 480, "Frame encoding error")
                ret, jpeg = cv2.imencode('.jpg', blank_frame)
                if not ret:  # If even this fails, return None
                    return None
                    
            return jpeg.tobytes()
        except Exception as e:
            print(f"Error encoding frame: {e}")
            # Create an error frame
            try:
                blank_frame = create_blank_frame(640, 480, f"Camera error: {str(e)[:30]}")
                ret, jpeg = cv2.imencode('.jpg', blank_frame)
                if ret:
                    return jpeg.tobytes()
            except:
                pass
            
            return None
    
    def get_recognition_result(self):
        """Return the last face recognition result."""
        return self.face_processor.last_recognition_result
    
    def get_face_thumbnail(self, thumbnail_id):
        """Return a specific face thumbnail as JPEG bytes.
        
        Args:
            thumbnail_id: Either a name (for known faces) or unique ID (for unknown faces)
            
        Returns:
            JPEG bytes of the thumbnail image
        """
        return self.face_processor.get_face_thumbnail(thumbnail_id)
    
    def toggle_recognition(self, enabled=None):
        """Toggle face recognition processing.
        
        Args:
            enabled: If provided, set to this value, otherwise toggle
            
        Returns:
            bool: Current recognition enabled state
        """
        if enabled is not None:
            self.recognition_enabled = enabled
        else:
            self.recognition_enabled = not self.recognition_enabled
            
        # If enabling recognition, reload faces from the database
        if self.recognition_enabled:
            self.face_processor.load_known_faces_from_db()
            
        return self.recognition_enabled
    
    def reset_recognition_state(self):
        """Reset the face recognition state."""
        self.face_processor.reset_state()
        self.process_this_frame = True
    
    def get_camera_properties(self):
        """Return properties of the camera.
        
        Returns:
            dict: Camera properties including resolution and FPS
        """
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


def list_available_cameras(max_cameras=8):
    """List all available cameras by efficiently testing common indices.
    
    Args:
        max_cameras: Maximum number of camera indices to test (default: 8)
        
    Returns:
        List of available camera indices
    """
    from app.camera.utils.camera_utils import list_available_cameras as list_cameras
    return list_cameras(max_cameras)
