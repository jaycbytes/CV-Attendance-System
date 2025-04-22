import cv2

def list_available_cameras(max_tested=8):
    """List all available cameras by efficiently testing common indices.
    
    This version is designed specifically to detect all cameras including 
    USB webcams on macOS and other operating systems.
    
    Args:
        max_tested: Maximum number of camera indices to test (default: 8)
        
    Returns:
        List of available camera indices
    """
    import time
    
    # List to store available cameras
    available_cameras = []
    
    start_time = time.time()
    print("Scanning for available cameras...")
    
    # Simple check that works well on macOS
    for i in range(max_tested):
        try:
            print(f"Checking camera {i}...")
            cap = cv2.VideoCapture(i)
            
            # Important: On macOS, a camera can be opened but not provide frames right away
            # So we'll consider any camera that can be opened as available
            if cap.isOpened():
                # Try to read a frame, but don't require success
                ret, frame = cap.read()
                
                # Add the camera if it can be opened, regardless of frame read success
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

def select_camera(cameras):
    print("Available Cameras:")
    for idx, cam in enumerate(cameras):
        print(f"{idx}: Camera {cam}")
    choice = int(input("Select a camera by number: "))
    return cameras[choice]

def open_camera_stream(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    print("Press 'q' to quit the camera view.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow(f"Live Feed - Camera {camera_index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    cameras = list_available_cameras()
    if not cameras:
        print("No cameras found.")
        return

    chosen_index = select_camera(cameras)
    open_camera_stream(chosen_index)

if __name__ == "__main__":
    main()
