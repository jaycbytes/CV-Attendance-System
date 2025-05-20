import cv2
import time
import numpy as np

def list_available_cameras(max_cameras=8):
    """List all available cameras by efficiently testing common indices.
    
    This version is designed specifically to detect all cameras including 
    USB webcams on macOS and other operating systems.
    
    Args:
        max_cameras: Maximum number of camera indices to test (default: 8)
        
    Returns:
        List of available camera indices
    """
    # List to store available cameras
    available_cameras = []
    
    start_time = time.time()
    print("Scanning for available cameras...")
    
    # First pass - simple check that works well on macOS
    for i in range(max_cameras):
        try:
            print(f"Checking camera {i}...")
            cap = cv2.VideoCapture(i)
            
            # Important: On macOS, a camera can be opened but not provide frames right away
            # We'll consider any camera that can be opened as available, even if it doesn't read a frame
            if cap.isOpened():
                # Try to read a frame, but don't require success
                ret, frame = cap.read()
                
                # Add the camera regardless of frame read success
                available_cameras.append(i)
                print(f"Camera {i} is available" + (" (read successful)" if ret else " (no frame read)"))
            else:
                print(f"Camera {i} failed to open")
                
            # Always release the camera
            cap.release()
            
        except Exception as e:
            print(f"Error checking camera {i}: {e}")
    
    # Print results
    elapsed = time.time() - start_time
    print(f"Camera scan completed in {elapsed:.2f} seconds")
    print(f"Found {len(available_cameras)} cameras: {available_cameras}")
    
    return available_cameras

def create_blank_frame(width=640, height=480, text=None):
    """Create a blank white frame with optional text.
    
    Args:
        width: Frame width
        height: Frame height
        text: Optional text to display on the frame
        
    Returns:
        Numpy array representing the frame
    """
    blank_frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    if text:
        cv2.putText(
            blank_frame, 
            text, 
            (width // 10, height // 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 0, 0), 
            2
        )
    
    return blank_frame

def try_camera_resolutions(camera, resolutions=None):
    """Try different common resolutions to find one that works with the camera.
    
    Args:
        camera: OpenCV VideoCapture object
        resolutions: List of (width, height) tuples to try
        
    Returns:
        (width, height) of the best working resolution
    """
    if resolutions is None:
        # Try common resolutions in order (from lower to higher)
        resolutions = [(640, 480), (800, 600), (1280, 720), (1920, 1080)]
    
    # Get default resolution
    default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Default camera resolution: {default_width}x{default_height}")
    
    # If default resolution seems valid, use it
    if default_width > 0 and default_height > 0:
        # Try to read a frame to validate, but don't require success
        ret, _ = camera.read()
        if ret:
            print(f"Successfully validated default resolution {default_width}x{default_height}")
        else:
            print(f"Default resolution seems valid but no frame read yet. Will still use {default_width}x{default_height}")
        
        return default_width, default_height
    
    # Otherwise try each resolution in the list
    best_width, best_height = 640, 480  # Default fallback
    
    for width, height in resolutions:
        print(f"Trying resolution {width}x{height}...")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Give camera a moment to adjust
        time.sleep(0.1)
        
        # Check if the setting worked
        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Actual resolution: {actual_width}x{actual_height}")
        
        # Try to read a frame, but don't strictly require success
        # Some USB cameras need more time to initialize
        ret, _ = camera.read()
        print(f"Frame read {'successful' if ret else 'unsuccessful'} at {actual_width}x{actual_height}")
        
        # If we got any dimensions, consider it working
        if actual_width > 0 and actual_height > 0:
            best_width, best_height = actual_width, actual_height
            # If frame read succeeded, that's the best case scenario
            if ret:
                break
    
    return best_width, best_height

def set_camera_mjpeg(camera):
    """Try to set camera to MJPEG mode for better compatibility with webcams.
    
    Args:
        camera: OpenCV VideoCapture object
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        camera.set(cv2.CAP_PROP_FOURCC, fourcc)
        print("Set camera format to MJPG")
    except Exception as e:
        print(f"Failed to set camera format: {e}")
