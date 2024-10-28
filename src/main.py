import cv2
import threading
import queue
import warnings
from database import init_db, mark_attendance
from face_recognition_module import load_known_faces, frame_processing
from gui_module import create_gui, update_gui

# Ignored this warning because it hasn't affected the program
warnings.filterwarnings(
    "ignore", message="AVCaptureDeviceTypeExternal is deprecated for Continuity Cameras"
)

face_images_dir = "../known_faces"


def main():
    # A function from the database.py file with simple logic to create our attendance db
    init_db()
    # This is where the images in known_faces folder are processed. !When testing make sure to change directory and provide at least one portrait image!
    known_faces = load_known_faces(face_images_dir)

    root, welcome_screen, video_label, welcome_label, face_image_label = create_gui()

    frame_queue = queue.Queue()
    video_capture = cv2.VideoCapture(0)

    # Set camera field of view (FOV)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Start a thread to process video frames
    threading.Thread(
        target=frame_processing,
        args=(video_capture, known_faces, frame_queue),
    ).start()

    # Start the GUI update loop and pass the known_faces directory. !Make sure to set your appropriate directory!
    root.after(
        0,
        update_gui,
        video_label,
        root,
        welcome_screen,
        welcome_label,
        face_image_label,
        frame_queue,
        face_images_dir,
    )
    root.mainloop()

    video_capture.release()


if __name__ == "__main__":
    main()
