import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps
import cv2
import os
from face_recognition_module import faces_dir

"""This file/modules job is to handle everything related to the Tkinter GUI.
This includes creating our windows and everything that goes inside such as labels.
Updating our Welcome screen and the GUI as a whole.
"""


def create_gui():
    """Create two windows for the GUI: one for the video feed and another for the welcome message."""
    root = tk.Tk()
    root.title("AI Club Attendance System")
    root.geometry("800x600")

    style = ttk.Style()
    style.configure(
        "TLabel",
        font=("Arial", 20),
        background="#ADD8E6",
    )
    style.configure(
        "bold.TLabel",
        font=("Helvetica", 24, "bold"),
        background="#ADD8E6",
        justify="center",
    )

    # Video feed label
    video_feed_label = ttk.Label(root)
    video_feed_label.pack()

    # Create a new window for the detected faces
    detected_faces_window = tk.Toplevel(root)
    detected_faces_window.title("Detected Faces")
    detected_faces_window.geometry("500x500")
    detected_faces_window.configure(bg="#ADD8E6")

    # Welcome screen window
    welcome_screen = tk.Toplevel(root)
    welcome_screen.title("Welcome Screen!")
    welcome_screen.geometry("500x500")
    welcome_screen.configure(bg="#ADD8E6")  # Light blue background color

    # Welcome label (text)
    welcome_label = ttk.Label(
        welcome_screen,
        text="Welcome to the club meeting!",
    )  # Background matching window
    welcome_label.pack(padx=10, pady=10)

    # Face image label (for showing recognized face)
    face_image_label = ttk.Label(welcome_screen)
    face_image_label.pack(pady=20)  # Adding padding for better layout

    return (
        root,
        welcome_screen,
        video_feed_label,
        welcome_label,
        face_image_label,
        detected_faces_window,
    )


def welcome_screen_update(face_name, welcome_label, face_image_label, face_images_dir):
    """Update the welcome screen with the recognized face name and display their profile image."""
    welcome_label.config(
        text=f"Welcome to the club meeting, {face_name}!",
        style="bold.TLabel",
    )

    # Try to find the image with any known extension
    possible_extensions = ["png", "jpg", "jpeg"]  # Possible image file extensions
    face_image_path = None

    # Look for the image file with one of the possible extensions
    for ext in possible_extensions:
        potential_path = os.path.join(face_images_dir, f"{face_name}.{ext}")
        if os.path.exists(potential_path):
            face_image_path = potential_path
            break

    if face_image_path:
        # Load the face image
        face_image = Image.open(face_image_path)

        # Maintain aspect ratio by fitting into a bounding box (e.g., 150x150)
        face_image = ImageOps.contain(
            face_image, (300, 300)
        )  # This keeps the aspect ratio intact

        face_image_tk = ImageTk.PhotoImage(face_image)

        face_image_label.config(image=face_image_tk)
        face_image_label.image = (
            face_image_tk  # Keep a reference to avoid garbage collection
        )
    else:
        # Display text if no image is found
        face_image_label.config(text="No profile image available")


def update_image_grid(
    detected_faces_window,
):
    if not hasattr(detected_faces_window, "images"):
        detected_faces_window.images = []

    images = []

    os.makedirs(faces_dir, exist_ok=True)

    for image_name in os.listdir(faces_dir):
        image_path = os.path.join(faces_dir, image_name)
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error: {e}")
            continue
        image = ImageOps.contain(image, (100, 100))
        image_tk = ImageTk.PhotoImage(image)
        images.append(image_tk)

    detected_faces_window.images = images

    width = detected_faces_window.winfo_width()
    height = detected_faces_window.winfo_height()
    if width <= 1 or height <= 1:
        detected_faces_window.after(1000, update_image_grid, detected_faces_window)
        return

    image_size = (100, 100)
    columns = width // image_size[0]

    if hasattr(detected_faces_window, "image_frame"):
        detected_faces_window.image_frame.destroy()

    frame = ttk.Frame(detected_faces_window)
    frame.grid(row=0, column=0)
    detected_faces_window.image_frame = frame

    image_labels = []

    for i, image in enumerate(images):
        row = i // columns
        column = i % columns
        label = ttk.Label(frame, image=image)
        label.grid(row=row, column=column)
        image_labels.append(label)

    detected_faces_window.after(3000, update_image_grid, detected_faces_window)


def update_gui(
    video_feed_label,
    root,
    welcome_screen,
    welcome_label,
    face_image_label,
    frame_queue,
    face_images_dir,
):
    """Update the GUI with the latest frame and recognized name."""

    # Check if there is a new frame available in the frame queue
    if not frame_queue.empty():
        # Retrieve the latest frame and the recognized face name from the queue
        frame, name = frame_queue.get()

        # OpenCV uses BGR format, but Tkinter's ImageTk uses RGBA, so we need to convert it
        cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Convert the frame into an ImageTk object that can be displayed in the Tkinter label
        img = ImageTk.PhotoImage(Image.fromarray(cv2_image))

        # This updates the Tkinter GUI to display the latest video frame from the camera
        video_feed_label.configure(image=img)

        # Keep a reference to avoid garbage collection
        video_feed_label.image = img

        # Update the wrap length of the welcome label based on the window size
        welcome_label.config(wraplength=welcome_screen.winfo_width() - 20)

        # If a face was recognized, update the welcome screen with the person's name and image
        if name:
            # Call the function to update the welcome screen with the recognized name and their profile picture
            welcome_screen_update(
                name, welcome_label, face_image_label, face_images_dir
            )

    # This creates a loop that keeps the GUI updated with the latest video frame and recognized names
    root.after(
        10,
        update_gui,
        video_feed_label,
        root,
        welcome_screen,
        welcome_label,
        face_image_label,
        frame_queue,
        face_images_dir,
    )
