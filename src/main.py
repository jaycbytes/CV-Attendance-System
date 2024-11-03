import cv2
import threading
import queue
import warnings
from database import init_db, mark_attendance
from face_recognition_module import (
    load_known_faces,
    frame_processing,
    initialize_hdf5_file,
    cluster_faces,
)
from gui_module import create_gui, update_gui, update_image_grid
from os.path import dirname, join as joinpath

# Ignored this warning because it hasn't affected the program
warnings.filterwarnings(
    "ignore", message="AVCaptureDeviceTypeExternal is deprecated for Continuity Cameras"
)

face_images_dir = joinpath(dirname(__file__), "../known_faces")


def main():
    # A function from the database.py file with simple logic to create our attendance db
    init_db()
    initialize_hdf5_file()

    # This is where the images in known_faces folder are processed. !When testing make sure to change directory and provide at least one portrait image!
    known_faces = load_known_faces(face_images_dir)

    (
        root,
        welcome_screen,
        video_label,
        welcome_label,
        face_image_label,
        detected_faces_window,
    ) = create_gui()

    frame_queue = queue.Queue()
    face_queue = queue.Queue()

    bounds = []

    video_capture = cv2.VideoCapture(0)

    # Set camera field of view (FOV)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Start a thread to process video frames
    threading.Thread(
        target=frame_processing,
        args=(
            video_capture,
            known_faces,
            frame_queue,
            face_queue,
            bounds,
        ),
    ).start()

    # Start a thread to cluster faces
    threading.Thread(
        target=cluster_faces,
        args=(face_queue,),
        daemon=True,
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
    root.after(
        0,
        update_image_grid,
        detected_faces_window,
    )
    root.mainloop()

    video_capture.release()


if __name__ == "__main__":
    main()
