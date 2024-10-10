import tkinter as tk
from PIL import Image, ImageTk, ImageOps
import cv2
import os
"""This file/modules job is to handle everything related to the Tkinter GUI.
This includes creating our windows and everything that goes inside such as labels.
Updating our Welcome screen and the GUI as a whole.
"""
def create_gui(): 
    """Create two windows for the GUI: one for the video feed and another for the welcome message."""
    root = tk.Tk()
    root.title("AI Club Attendance System")
    root.geometry("800x600")

    # Video feed label
    video_feed_label = tk.Label(root)
    video_feed_label.pack()

    # Welcome screen window
    welcome_screen = tk.Toplevel(root)
    welcome_screen.title("Welcome Screen!")
    welcome_screen.geometry("500x500")
    welcome_screen.configure(bg='#ADD8E6')  # Light blue background color

    # Welcome label (text)
    welcome_label = tk.Label(welcome_screen, text="Welcome to the club meeting!", 
                             font=("Arial", 20), bg='#ADD8E6')  # Background matching window
    welcome_label.pack(padx=10, pady=10)

    # Face image label (for showing recognized face)
    face_image_label = tk.Label(welcome_screen, bg='#ADD8E6')
    face_image_label.pack(pady=20)  # Adding padding for better layout

    return root, video_feed_label, welcome_label, face_image_label

def welcome_screen_update(face_name, welcome_label, face_image_label, face_images_dir):
    """Update the welcome screen with the recognized face name and display their profile image."""
    welcome_label.config(text=f"Welcome to the club meeting, {face_name}!", font=("Helvetica", 24, "bold"))

    # Try to find the image with any known extension
    possible_extensions = ['png', 'jpg', 'jpeg']  # Possible image file extensions
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
        face_image = ImageOps.contain(face_image, (300, 300))  # This keeps the aspect ratio intact

        face_image_tk = ImageTk.PhotoImage(face_image)

        face_image_label.config(image=face_image_tk)
        face_image_label.image = face_image_tk  # Keep a reference to avoid garbage collection
    else:
        # Display text if no image is found
        face_image_label.config(text="No profile image available", font=("Helvetica", 14, "italic"))

def update_gui(video_feed_label, root, welcome_label, face_image_label, frame_queue, face_images_dir):
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

        # If a face was recognized, update the welcome screen with the person's name and image
        if name:
            # Call the function to update the welcome screen with the recognized name and their profile picture
            welcome_screen_update(name, welcome_label, face_image_label, face_images_dir)
            
    # This creates a loop that keeps the GUI updated with the latest video frame and recognized names
    root.after(10, update_gui, video_feed_label, root, welcome_label, face_image_label, frame_queue, face_images_dir)
