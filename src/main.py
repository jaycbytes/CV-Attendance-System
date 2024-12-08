import sys
import cv2
import numpy as np
import os
import face_recognition
import json
import time
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
    QWidget, QSplitter, QScrollArea, QGridLayout, QDialog,
    QLineEdit, QPushButton, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer, QSize
from collections import deque
from datetime import datetime
from face_recognition_module import (
    load_known_faces, initialize_face_encodings, 
    assess_face_quality, compare_face_similarity
)

# Configure face storage
FACES_DIR = os.path.abspath("known_faces")
ENCODINGS_FILE = os.path.join(FACES_DIR, "face_encodings.json")

class RegistrationDialog(QDialog):
    def __init__(self, face_image, face_encoding, parent=None):
        super().__init__(parent)
        self.face_image = face_image
        self.face_encoding = face_encoding
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Member Registration")
        self.setFixedWidth(400)
        layout = QVBoxLayout(self)

        # Convert face image for display
        h, w = self.face_image.shape[:2]
        bytes_per_line = 3 * w
        rgb_image = cv2.cvtColor(self.face_image, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(
            150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Face preview
        face_label = QLabel()
        face_label.setPixmap(pixmap)
        face_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(face_label)

        # Name input
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name")
        self.name_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #4a4a4a;
                border-radius: 5px;
                background-color: #2b2b2b;
                color: white;
            }
        """)
        layout.addWidget(self.name_input)

        # Major input
        self.major_input = QLineEdit()
        self.major_input.setPlaceholderText("Enter major")
        self.major_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #4a4a4a;
                border-radius: 5px;
                background-color: #2b2b2b;
                color: white;
            }
        """)
        layout.addWidget(self.major_input)

        # Don't ask again checkbox
        self.dont_ask_checkbox = QCheckBox("Don't ask me again this session")
        self.dont_ask_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                padding: 5px;
            }
        """)
        layout.addWidget(self.dont_ask_checkbox)

        # Buttons
        button_layout = QHBoxLayout()

        self.save_button = QPushButton("Save")
        self.save_button.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                background-color: #4CAF50;
                border: none;
                border-radius: 5px;
                color: white;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.delete_button = QPushButton("Delete")
        self.delete_button.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                background-color: #f44336;
                border: none;
                border-radius: 5px;
                color: white;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                background-color: #4a4a4a;
                border: none;
                border-radius: 5px;
                color: white;
            }
            QPushButton:hover {
                background-color: #585858;
            }
        """)

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.save_button.clicked.connect(self.accept)
        self.delete_button.clicked.connect(self.reject_with_delete)
        self.cancel_button.clicked.connect(self.reject)

        # Dialog styling
        self.setStyleSheet("""
            QDialog {
                background-color: #3a3a3a;
            }
            QLabel {
                color: white;
            }
        """)

    def reject_with_delete(self):
        self.done(QDialog.Rejected)
        self.deleteLater()

class VideoThread(QThread):
    frame_processed = pyqtSignal(np.ndarray, str, str)  # frame, name, major
    new_face_detected = pyqtSignal(np.ndarray, np.ndarray)  # face_image, face_encoding

    def __init__(self):
        super().__init__()
        self._running = True
        print("Loading known faces...")
        self.known_faces = load_known_faces(FACES_DIR)
        print(f"Loaded {len(self.known_faces)} known faces")
        for name in self.known_faces:
            print(f"Loaded face for: {name}")
        self.mutex = QMutex()
        self.frame_skip = 2  # Process every nth frame
        self.frame_count = 0
        self.last_detection = None
        self.last_detection_time = {}  # Store last detection time for each face

    def run(self):
        print("Starting camera capture...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("Camera opened successfully")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                continue

            # Always emit original frame first for display
            self.frame_processed.emit(frame.copy(), "", "")

            # Process every nth frame for face detection
            if self.frame_count % self.frame_skip == 0:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame = np.ascontiguousarray(rgb_frame)

                    # Detect faces
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                    
                    if face_locations:
                        self.last_detection = {
                            'locations': face_locations,
                            'frame': frame.copy()
                        }
                        
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                            name = "Unknown"
                            major = "Unknown"
                            
                            face_image = frame[top:bottom, left:right].copy()
                            quality_score = assess_face_quality(face_image)
                            
                            # Skip low quality faces
                            if quality_score < 30:  # Adjust threshold as needed
                                continue
                            
                            # Check cooldown for unknown faces
                            face_hash = hash(encoding.tobytes())
                            current_time = time.time()
                            if face_hash in self.last_detection_time:
                                if current_time - self.last_detection_time[face_hash] < 5:  # 5 second cooldown
                                    continue
                            
                            # Check against known faces
                            self.mutex.lock()
                            try:
                                for known_name, known_data in self.known_faces.items():
                                    if compare_face_similarity(known_data['encoding'], encoding):
                                        name = known_name
                                        major = known_data['major']
                                        print(f"Recognized face: {name}")
                                        break
                            finally:
                                self.mutex.unlock()
                                
                            # Draw rectangle around face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(frame, name, (left, bottom + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # If unknown face, emit signal for clustering
                            if name == "Unknown":
                                self.last_detection_time[face_hash] = current_time
                                self.new_face_detected.emit(face_image, encoding)

                            # Emit the processed frame with recognition info
                            self.frame_processed.emit(frame.copy(), name, major)

                except Exception as e:
                    print(f"Error processing frame: {e}")

            elif self.last_detection:
                # Draw the last known rectangles on non-processing frames
                frame_copy = frame.copy()
                for (top, right, bottom, left) in self.last_detection['locations']:
                    cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 2)
                self.frame_processed.emit(frame_copy, "", "")
            else:
                # Always emit a frame for display
                self.frame_processed.emit(frame.copy(), "", "")

            self.frame_count += 1

        cap.release()
        print("Camera released")

    def stop(self):
        self._running = False
        self.wait()

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setStyleSheet("""
            QLabel {
                border: 2px solid transparent;
                border-radius: 5px;
                padding: 2px;
                margin: 1px;
            }
            QLabel:hover {
                border-color: #4CAF50;
            }
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

class WelcomeQueueWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(60)
        self.setStyleSheet("""
            QWidget {
                background-color: #3a3a3a;
                border-radius: 10px;
                border: 2px solid #4a4a4a;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
        """)
        
        # Create layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 5, 10, 5)
        self.layout.setSpacing(5)
        
        # Queue label with fixed width
        self.label = QLabel("Queue:")
        self.label.setStyleSheet("font-size: 14px;")
        self.label.setFixedWidth(45)  # Fixed compact width
        self.layout.addWidget(self.label)
        
        # Container for member thumbnails
        self.queue_container = QWidget()
        self.queue_layout = QHBoxLayout(self.queue_container)
        self.queue_layout.setContentsMargins(0, 0, 0, 0)
        self.queue_layout.setSpacing(5)
        self.queue_layout.setAlignment(Qt.AlignLeft)  # Align thumbnails to the left
        self.layout.addWidget(self.queue_container, stretch=1)  # Give more space to container
        
        self.thumbnail_size = QSize(40, 40)
        self.thumbnails = []

    def update_queue(self, queue_data):
        # Clear existing thumbnails
        for thumbnail in self.thumbnails:
            self.queue_layout.removeWidget(thumbnail)
            thumbnail.deleteLater()
        self.thumbnails.clear()
        
        # Add new thumbnails
        for name, photo_path in queue_data:
            if photo_path and os.path.exists(photo_path):
                container = QWidget()
                container.setFixedSize(self.thumbnail_size)
                
                # Member photo
                photo_label = QLabel(container)
                photo_label.setFixedSize(self.thumbnail_size)
                pixmap = QPixmap(photo_path)
                scaled_pixmap = pixmap.scaled(
                    self.thumbnail_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                photo_label.setPixmap(scaled_pixmap)
                photo_label.setStyleSheet("""
                    QLabel {
                        border: 1px solid #4a4a4a;
                        border-radius: 5px;
                    }
                """)
                
                self.queue_layout.addWidget(container)
                self.thumbnails.append(container)





class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Club Attendance System")
        self.setGeometry(100, 100, 1200, 800)
        
        self.member_queue = deque()
        self.current_display_timer = QTimer()
        self.current_display_timer.timeout.connect(self.process_queue)
        self.display_duration = 7000  # 7 seconds in milliseconds
        # Track opted-out faces
        self.opted_out_faces = set()
        

        

        # Create main layout
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #2b2b2b;")  # Dark background
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left side - Video and Welcome Card
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Welcome Card with updated style
        self.welcome_card = QWidget()
        self.welcome_card.setStyleSheet("""
            QWidget {
                background-color: #3a3a3a;
                border-radius: 10px;
                border: 2px solid #4a4a4a;
            }
            QLabel {
                color: #ffffff;
                font-size: 24px;
            }
        """)
        welcome_layout = QHBoxLayout(self.welcome_card)

        self.queue_display = WelcomeQueueWidget()
        left_layout.addWidget(self.queue_display)
        
        # Left side of welcome card - Text information
        self.welcome_text = QLabel("Welcome!")
        self.welcome_text.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(self.welcome_text, stretch=1)
        
        # Right side of welcome card - Member photo
        self.member_photo = QLabel()
        self.member_photo.setFixedSize(250, 250)  # Fixed size for photo
        self.member_photo.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.member_photo.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(self.member_photo, alignment=Qt.AlignCenter)
        
        left_layout.addWidget(self.welcome_card, stretch=2)
        
        # Video Feed with frame - smaller size
        video_container = QWidget()
        video_container.setStyleSheet("""
            QWidget {
                background-color: #3a3a3a;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        video_layout = QVBoxLayout(video_container)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(480, 360)  # Reduced size
        self.video_label.setStyleSheet("background-color: #2b2b2b;")
        video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        left_layout.addWidget(video_container, stretch=1)
        
        main_layout.addWidget(left_widget)
        
        # Right side - Unrecognized Faces
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setMaximumWidth(300)
        right_widget.setStyleSheet("""
            QWidget {
                background-color: #3a3a3a;
                border-radius: 10px;
            }
        """)
        
        # Title for unrecognized faces section
        title_label = QLabel("Unrecognized Faces")
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 18px;
                padding: 5px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title_label)
        
        # Scroll area for unrecognized faces
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #2b2b2b;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #4a4a4a;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        self.faces_widget = QWidget()
        self.faces_widget.setStyleSheet("background-color: transparent;")
        self.faces_layout = QGridLayout(self.faces_widget)
        self.faces_layout.setSpacing(2)  # Reduced spacing
        self.faces_layout.setContentsMargins(2, 2, 2, 2)  # Reduced margins
        scroll.setWidget(self.faces_widget)
        right_layout.addWidget(scroll)
        
        main_layout.addWidget(right_widget)
        
        main_layout.addWidget(right_widget)
        
        # Store unrecognized faces
        self.unrecognized_faces = []
        
        # Start video thread
        self.video_thread = VideoThread()
        self.video_thread.frame_processed.connect(self.update_frame)
        self.video_thread.new_face_detected.connect(self.add_unrecognized_face)
        self.video_thread.start()

    def update_frame(self, frame, name, major):
        try:
            # Convert frame to QImage
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Update video feed
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                480, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
            
            # Handle recognized faces without interrupting current display
            if name and name != "Unknown":
                self.add_to_queue(name, major)
                
        except Exception as e:
            print(f"Error updating frame: {e}")
        

    def add_unrecognized_face(self, face_image, face_encoding):
        # Strict quality check first
        quality_score = assess_face_quality(face_image)
        if quality_score < 50:  # Increased quality threshold
            print(f"Rejecting face with low quality score: {quality_score}")
            return

        # Check against known faces with very lenient tolerance
        self.video_thread.mutex.lock()
        try:
            for known_name, known_data in self.video_thread.known_faces.items():
                # Use a more lenient tolerance for known faces to ensure we don't add them
                if face_recognition.compare_faces([known_data['encoding']], face_encoding, tolerance=0.65)[0]:
                    print(f"Face matches known member {known_name}, not adding to unknown faces")
                    return
        finally:
            self.video_thread.mutex.unlock()

        # Check against existing unrecognized faces with strict tolerance
        for i, existing_face in enumerate(self.unrecognized_faces):
            if compare_face_similarity(existing_face['encoding'], face_encoding):
                # If new face is higher quality, replace the old one
                existing_quality = assess_face_quality(existing_face['image'])
                if quality_score > existing_quality + 10:  # Only replace if significantly better
                    print(f"Replacing face (quality: {existing_quality} -> {quality_score})")
                    existing_face['image'] = face_image
                    existing_face['encoding'] = face_encoding
                    existing_face['quality'] = quality_score
                    self.update_unrecognized_faces_display()
                return
        
        # If we get here, it's a new unique face
        print(f"Adding new unique face with quality score: {quality_score}")
        self.unrecognized_faces.append({
            'image': face_image,
            'encoding': face_encoding,
            'quality': quality_score,
            'added_time': time.time()
        })
        
        # Keep only the highest quality faces, prefer newer faces when quality is similar
        max_faces = 12
        if len(self.unrecognized_faces) > max_faces:
            # Sort by quality and recency
            current_time = time.time()
            self.unrecognized_faces.sort(
                key=lambda x: (x['quality'] * 0.8 + 
                             (20 * (1 - min(1, (current_time - x['added_time']) / 300)))),
                reverse=True
            )
            self.unrecognized_faces = self.unrecognized_faces[:max_faces]
        
        self.update_unrecognized_faces_display()

        
    def update_unrecognized_faces_display(self):
        # Clear existing layout
        for i in reversed(range(self.faces_layout.count())): 
            self.faces_layout.itemAt(i).widget().setParent(None)
            
        # Add faces to grid
        for i, face_data in enumerate(self.unrecognized_faces):
            face_image = face_data['image']
            
            # Convert to QImage
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(
                80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation  # Reduced size from 100x100 to 80x80
            )
            
            # Create clickable label for face
            label = ClickableLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            
            # Store face data with the label
            label.face_image = face_image
            label.face_encoding = face_data['encoding']
            
            # Connect click handler
            label.clicked.connect(lambda l=label: self.handle_unknown_face_click(l))
            
            # Add to grid layout
            row = i // 3  # Changed from 2 to 3 faces per row
            col = i % 3
            self.faces_layout.addWidget(label, row, col)

    def handle_unknown_face_click(self, label):
        # Check if face is opted out
        face_hash = hash(label.face_encoding.tobytes())
        if face_hash in self.opted_out_faces:
            return

        # Show registration dialog
        dialog = RegistrationDialog(label.face_image, label.face_encoding, self)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            name = dialog.name_input.text().strip()
            major = dialog.major_input.text().strip()

            if not name or not major:
                QMessageBox.warning(self, "Input Error", "Both name and major are required.")
                return

            try:
                # Save face image
                image_path = os.path.join(FACES_DIR, f"{name}.png")
                cv2.imwrite(image_path, label.face_image)

                # Update face encodings
                self.video_thread.mutex.lock()
                try:
                    self.video_thread.known_faces[name] = {
                        'encoding': label.face_encoding,
                        'major': major
                    }

                    # Load existing known faces
                    known_faces = {}
                    if os.path.exists(ENCODINGS_FILE):
                        try:
                            with open(ENCODINGS_FILE, 'r') as f:
                                known_faces = json.load(f)
                        except Exception as e:
                            print(f"Error loading existing known faces: {e}")

                    # Add new member to known faces
                    known_faces[name] = {
                        'encoding': label.face_encoding.tolist(),
                        'major': major
                    }

                    # Save updated known faces
                    try:
                        with open(ENCODINGS_FILE, 'w') as f:
                            json.dump(known_faces, f, indent=4)
                        print(f"Successfully saved encodings for {name}")
                    except Exception as e:
                        print(f"Error saving encodings for {name}: {e}")

                finally:
                    self.video_thread.mutex.unlock()

                print(f"Successfully registered new member: {name}")

                # Remove face from unrecognized faces
                self.unrecognized_faces = [
                    face for face in self.unrecognized_faces
                    if not compare_face_similarity(face['encoding'], label.face_encoding)
                ]
                self.update_unrecognized_faces_display()

            except Exception as e:
                print(f"Error registering new member: {e}")
                QMessageBox.critical(self, "Error", "Failed to save member information.")

        elif result == QDialog.Rejected:
            # User clicked the "Delete" button
            # Remove face from unrecognized faces
            self.unrecognized_faces = [
                face for face in self.unrecognized_faces
                if not compare_face_similarity(face['encoding'], label.face_encoding)
            ]
            self.update_unrecognized_faces_display()

        elif dialog.dont_ask_checkbox.isChecked():
            # Add to opted out faces
            self.opted_out_faces.add(face_hash)

    
    def add_to_queue(self, name, major):
        """Add a member to the welcome queue if not already present."""
        # Check if member is already in queue or currently displayed
        if not any(member[0] == name for member in self.member_queue):
            photo_path = os.path.join(FACES_DIR, f"{name}.png")
            
            # If no one is being displayed, start with this member
            if not self.current_display_timer.isActive():
                self.member_queue.append((name, major, photo_path))
                self.display_current_member()
                self.current_display_timer.start(self.display_duration)
            else:
                # Add to queue without interrupting current display
                self.member_queue.append((name, major, photo_path))
            
            # Update queue display
            self.update_queue_display()

    def process_queue(self):
        """Process the next member in the queue."""
        if self.member_queue:
            # Remove current member
            current_name = self.member_queue[0][0]
            self.member_queue.popleft()
            
            if self.member_queue:
                # Display next member
                self.display_current_member()
                self.current_display_timer.start(self.display_duration)
            else:
                # Queue is empty, stop timer
                self.current_display_timer.stop()
                # Keep current display until timeout
                
        self.update_queue_display()

    def display_current_member(self):
        """Display the current member in the welcome card."""
        if self.member_queue:
            name, major, photo_path = self.member_queue[0]
            self.welcome_text.setText(f"Welcome {name}!\nMajor: {major}")
            
            if os.path.exists(photo_path):
                member_pixmap = QPixmap(photo_path)
                member_pixmap = member_pixmap.scaled(
                    250, 250,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.member_photo.setPixmap(member_pixmap)

    def update_queue_display(self):
        """Update the queue display widget."""
        queue_data = [(name, photo_path) for name, _, photo_path in self.member_queue]
        self.queue_display.update_queue(queue_data)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    os.makedirs(FACES_DIR, exist_ok=True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())