import cv2

def list_available_cameras(max_tested=10):
    available_cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available_cameras.append(i)
        cap.release()
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
