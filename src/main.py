import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QSplitter, QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QTabWidget, QGridLayout, QFormLayout, QLineEdit, QSizePolicy, QMessageBox, QDialog, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import sys
import cv2
import numpy as np
import queue
from cv2 import cvtColor, COLOR_BGR2RGB
from PIL import UnidentifiedImageError, Image, ImageOps, ImageDraw
import face_recognition
from face_recognition_module import clustering, height, width, draw_box, dbscan_predict, cluster_embeddings, initialize_hdf5_file
import h5py
from sklearn.cluster import DBSCAN
from sklearn import decomposition
from scipy.spatial.distance import cdist
from os.path import dirname, join as joinpath
import os
import time
import math
import json
import shutil
import re


data_file = joinpath(dirname(__file__), "../data")

face_encodings_dir = joinpath(data_file, "face_encodings.json")

database_file = joinpath(data_file, "database.h5")

faces_dir = joinpath(data_file, "faces")

known_faces_dir = joinpath(data_file, "known_faces")

names_data = joinpath(data_file, "names.json")

embeddings_data = joinpath(data_file, "embeddings.json")

def frame_processing(
    frame,
    face_queue,
    bounds,
    frame_count,
    process_every_n_frames=3,
):
    if (
        frame_count % process_every_n_frames == 0
    ):  # if the remainder isn't 0 then it's not a frame we want to process
        # Convert the frame to RGB
        rgb_frame = cvtColor(frame, COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        # Find all the face locations and encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        name = None

        bounds.clear()

        # Loop over each face found in the frame to see if it's someone we know
        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            # Create a small image of just the face
            cropped_face = image.crop((left, top, right, bottom))
            resized_face = cropped_face.resize((width, height))
            face = np.array(resized_face)

            face_queue.put((face, face_encoding))

            index = None

            if clustering is not None:
                index = dbscan_predict(clustering, [face_encoding])[0]

            try:
                if os.path.exists(names_data):
                    with open(names_data, "r") as f:
                        names = json.load(f)
                else:
                    names = {}

                known_faces = list(names.items())
                known_face_encodings = [face["embedding"] for _, face in known_faces]

                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding, tolerance=0.6
                )
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_faces[best_match_index]

            except Exception as e:
                print(e)

            bounds.append(((top, right, bottom, left), name))

        for bound, name in bounds:
            try:
                draw_box(frame, bound, name[0])
            except Exception as e:
                pass

        # Put the frame and the recognized name (if any) into the queue
        return (frame, name)
    else:
        for bound, name in bounds:
            try:
                draw_box(frame, bound, name[0])
            except Exception as e:
                pass

        # If not processing for recognition, still put the frame in the queue
        return (frame, None)


# https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, tuple)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.face_queue = queue.Queue()
        self.bounds = []
        self.frame_count = 0
        self.last_name = None

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                frame, name = frame_processing(
                    cv_img,
                    self.face_queue,
                    self.bounds,
                    self.frame_count,
                )
                if name is None:
                    name = self.last_name
                if name is None:
                    name = ("", {})
                self.change_pixmap_signal.emit(frame, name)
                self.frame_count += 1

        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

def cluster_faces(face_queue):
    global clustering

    with h5py.File(database_file, "a") as file:
        try:
            images = file["images"]
            embeddings = file["embeddings"]
        except KeyError:
            print("No images or embeddings found in database")
            initialize_hdf5_file()
            return

        for face, face_encoding in iter(face_queue.get, None):
            images.resize(images.shape[0] + 1, axis=0)
            images[-1] = face

            embeddings.resize(embeddings.shape[0] + 1, axis=0)
            embeddings[-1] = face_encoding

            if face_queue.qsize() == 0:
                break

        medoid_indices, clustering = cluster_embeddings(embeddings)

        medoid_images = images[sorted(medoid_indices)]
        medoid_embeddings = embeddings[sorted(medoid_indices)]

        try:
            os.makedirs(faces_dir, exist_ok=True)

            for file in os.listdir(faces_dir):
                os.remove(os.path.join(faces_dir, file))

            for i, medoid_image in enumerate(medoid_images):
                cv2.imwrite(
                    os.path.join(faces_dir, f"face_{i}.png"),
                    cvtColor(medoid_image, cv2.COLOR_RGB2BGR),
                )

            with open(embeddings_data, "w") as f:
                json.dump(medoid_embeddings.tolist(), f)
        except Exception as e:
            print(e)

class ClusterThread(QThread):
    clustered = pyqtSignal()

    def __init__(self, face_queue):
        super().__init__()
        self.face_queue = face_queue
        self._run_flag = True

        initialize_hdf5_file()

    def run(self):
        while self._run_flag:
            cluster_faces(self.face_queue)
            self.clustered.emit()
            for _ in range(50):
                if not self._run_flag:
                    break
                QThread.msleep(100)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit()

class ImageGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.image_labels = []
        self.layout = QVBoxLayout()
        self.grid_layout = QGridLayout()
        self.layout.addLayout(self.grid_layout)
        self.setLayout(self.layout)
        self.update_image_grid()

    def update_image_grid(self):
        os.makedirs(faces_dir, exist_ok=True)

        self.image_labels = []

        # Load the images from the faces directory
        for image_name in os.listdir(faces_dir):
            image_path = os.path.join(faces_dir, image_name)
            try:
                pixmap = QPixmap(image_path)
                image_label = ClickableLabel()
                image_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                image_label.clicked.connect(lambda path=image_path: self.open_face_dialog(path))
                self.image_labels.append(image_label)
            except Exception as e:
                print(f"Error: {e}")
                continue

        # Clear the layout
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        window_width = self.size().width()

        columns = max(1, math.floor(window_width / 100))
        rows = math.ceil(len(self.image_labels) / columns)

        self.grid_layout.setVerticalSpacing(0)
        self.grid_layout.setHorizontalSpacing(0)

        row = 0
        col = 0

        for idx, label in enumerate(self.image_labels):
            self.grid_layout.addWidget(label, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1

        total_height = rows * 100
        self.setMinimumHeight(total_height)
        self.setMaximumHeight(total_height)

    def open_face_dialog(self, image_path):
        dialog = FaceDialog(image_path)
        dialog.exec_()

    def resizeEvent(self, event):
        window_width = self.size().width()
        columns = max(1, math.floor(window_width / 100))
        rows = math.ceil(len(self.image_labels) / columns)

        for i, label in enumerate(self.image_labels):
            row = i // columns
            col = i % columns
            self.grid_layout.addWidget(label, row, col)

        total_height = rows * 100
        self.setMinimumHeight(total_height)
        self.setMaximumHeight(total_height)

        super().resizeEvent(event)

class FaceDialog(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle('Input face data')
        super().setModal(True)

        self.image_path = image_path
        self.layout = QFormLayout()

        # Display the image
        self.image = QLabel()
        pixmap = QPixmap(image_path)
        self.image.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.layout.addRow(self.image)

        # Add the name and major fields
        self.name = QLineEdit()
        self.layout.addRow('Name:', self.name)

        self.major = QLineEdit()
        self.layout.addRow('Major:', self.major)

        # Add the file button
        self.file_button = QtWidgets.QPushButton('Select Image (optional)')
        self.file_button.clicked.connect(self.get_file)
        self.layout.addRow(self.file_button)
        self.custom_image_path = None

        self.setLayout(self.layout)

        # Add the OK and Cancel buttons
        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout.addRow(self.buttonBox)

    def accept(self):
        name = self.name.text()
        major = self.major.text()

        if name == '':
            QMessageBox.warning(self, 'Warning', 'Name cannot be empty')
            return

        # Copy the image to the faces directory
        image_name = f"{name}.png"
        image_path = os.path.join(known_faces_dir, image_name)
        os.makedirs(known_faces_dir, exist_ok=True)

        face_num = int(re.findall(r'\d+', self.image_path)[-1])

        shutil.copyfile(self.image_path, image_path)

        with open(embeddings_data, 'r') as f:
            embeddings = json.load(f)

        if os.path.exists(names_data):
            with open(names_data, 'r') as f:
                names = json.load(f)
        else:
            names = {}

        names[name] = {
            'major': major,
            'image': image_path,
            'embedding': embeddings[face_num]
        }

        if self.custom_image_path is not None:
            names[name]['custom_image'] = self.custom_image_path

        with open(names_data, 'w') as f:
            json.dump(names, f)

        super().accept()

    def reject(self):
        super().reject()

    def get_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png)")
        self.custom_image_path = fname[0]
        self.image.setPixmap(QPixmap(fname[0]).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.file_button.setText('Change Image')


class WelcomeScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.name = None
        self.image_file = None

        self.welcome_label = QLabel('Welcome!')
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setStyleSheet('font-size: 24px;')
        self.layout.addWidget(self.welcome_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.setLayout(self.layout)

    def set_name(self, name):
        self.name = name
        self.welcome_label.setText(f'Welcome, {name}!')

    def set_image(self, image_file):
        self.image_file = image_file
        pixmap = QPixmap(image_file)
        self.image_label.setPixmap(pixmap)

# https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt live label demo")

        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)

        self.tabs = QTabWidget()
        self.welcome_screen = WelcomeScreen()
        self.tabs.addTab(self.welcome_screen, 'Welcome')
        self.image_grid = ImageGrid()
        self.tabs.addTab(self.image_grid, 'Detected Faces')

        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(0, 0)
        self.tabs.setMinimumSize(0, 0)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.image_label)
        self.splitter.addWidget(self.tabs)
        self.splitter.setCollapsible(0, True)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setOpaqueResize(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.splitter)
        self.setLayout(self.layout)
        self.splitter.setMaximumHeight(self.height())

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        self.cluster_thread = ClusterThread(self.thread.face_queue)
        self.cluster_thread.start()
        self.cluster_thread.clustered.connect(self.image_grid.update_image_grid)

    def closeEvent(self, event):
        self.thread.stop()
        self.cluster_thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray, tuple)
    def update_image(self, cv_img, name):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

        if name == ('', {}) or name[0] is None:
            return
        self.welcome_screen.set_name(name[0])
        self.welcome_screen.set_image(name[1]['custom_image'] if 'custom_image' in name[1] else name[1]['image'])

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        width = self.image_label.size().width()
        height = self.image_label.size().height()
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def set_max_height(self):
        # Set the label's max height to match the window's height
        self.splitter.setMaximumHeight(self.height())

    def resizeEvent(self, event):
        self.set_max_height()
        super().resizeEvent(event)

    def shutprocess(self):
        self.thread.stop()
        self.cluster_thread.stop()
        self.close()

if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
