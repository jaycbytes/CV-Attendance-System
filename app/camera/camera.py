import cv2
import threading

class Camera:
    """Base camera class for accessing webcam."""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.camera = None
        self.thread = None
        self.frame = None
        self.stopped = False
        
    def start(self):
        """Start the camera and capture thread."""
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera with ID {self.camera_id}")
        
        # Start the thread to read frames
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        return self
        
    def _capture_loop(self):
        """Capture frames in a loop."""
        while not self.stopped:
            success, frame = self.camera.read()
            if not success:
                break
            self.frame = frame
            
    def get_frame(self):
        """Convert frame to JPEG for MJPEG streaming."""
        if self.frame is None:
            return None
            
        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', self.frame)
        if not ret:
            return None
            
        return jpeg.tobytes()
        
    def stop(self):
        """Stop the camera thread and release resources."""
        self.stopped = True
        if self.thread:
            self.thread.join()
        if self.camera:
            self.camera.release()