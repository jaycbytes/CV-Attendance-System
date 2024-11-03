import face_recognition
import os
import numpy as np
from cv2 import cvtColor, COLOR_BGR2RGB
from PIL import UnidentifiedImageError, Image, ImageOps, ImageDraw
import json
from os.path import dirname, join as joinpath
import h5py
from sklearn.cluster import DBSCAN
from sklearn import decomposition
from scipy.spatial.distance import cdist
import cv2
import time
import scipy as sp

data_file = joinpath(dirname(__file__), "../data")

face_encodings_dir = joinpath(data_file, "face_encodings.json")

database_file = joinpath(data_file, "database.h5")

height = 640
width = 640

faces_dir = joinpath(data_file, "faces")

clustering = None


def initialize_hdf5_file():
    if not os.path.exists(database_file):
        os.makedirs(data_file, exist_ok=True)

        with h5py.File(database_file, "w") as file:
            # Create an images dataset
            file.create_dataset(
                "images",
                (0, height, width, 3),
                maxshape=(None, height, width, 3),
                dtype="uint8",
                # compression="gzip",
            )

            # Create an embeddings dataset
            file.create_dataset(
                "embeddings",
                (0, 128),
                maxshape=(None, 128),
                dtype="float32",
                # compression="gzip",
            )


# Load known faces of members from the images folder in order to encode and get names of each face
def load_known_faces(face_images_dir):
    known_faces = {}

    if not os.path.exists(face_encodings_dir):
        with open(face_encodings_dir, "w") as file:
            json.dump({}, file)

    with open(face_encodings_dir, "r") as file:
        known_faces = json.load(file)
        known_faces = {k: np.array(v) for k, v in known_faces.items()}

    # Loop through all images in the provided face_images_dir,
    for image_name in os.listdir(face_images_dir):
        image_path = os.path.join(face_images_dir, image_name)
        # The reason we have to use load_image_file is to convert our images into numpy arrays which face_recognition library requires
        face_name = os.path.splitext(image_name)[0]

        if face_name in known_faces:
            continue

        try:
            image = face_recognition.load_image_file(
                image_path
            )  # images can be .png or .jpg, other ext haven't been tested
        except UnidentifiedImageError:
            continue

        image_encodings = face_recognition.face_encodings(image)

        if image_encodings:
            known_faces[face_name] = image_encodings[0]
        else:
            print(f"Error: No face found in {image_name}")

    with open(face_encodings_dir, "w") as file:
        json_faces = {k: v.tolist() for k, v in known_faces.items()}
        json.dump(json_faces, file)

    return known_faces


def cluster_embeddings(embeddings, threshold=0.5):
    clustering = DBSCAN(eps=threshold, min_samples=2).fit(embeddings)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters)
    print("Estimated number of noise points: %d" % n_noise)

    medoid_indices = []

    for label in set(clustering.labels_):
        if label == -1:
            continue

        cluster_indices = np.nonzero(clustering.labels_ == label)[0]
        cluster_points = embeddings[cluster_indices]

        pairwise_distances = cdist(cluster_points, cluster_points, metric="euclidean")

        avg_distances = np.mean(pairwise_distances, axis=1)

        medoid_index = cluster_indices[np.argmin(avg_distances)]
        medoid_indices.append(medoid_index)

    return medoid_indices, clustering


def cluster_faces(face_queue):
    global clustering

    while True:

        with h5py.File(database_file, "a") as file:
            images = file["images"]
            embeddings = file["embeddings"]

            for face, face_encoding in iter(face_queue.get, None):
                images.resize(images.shape[0] + 1, axis=0)
                images[-1] = face

                embeddings.resize(embeddings.shape[0] + 1, axis=0)
                embeddings[-1] = face_encoding

                if face_queue.qsize() == 0:
                    break

            medoid_indices, clustering = cluster_embeddings(embeddings)

            medoid_images = images[sorted(medoid_indices)]

            try:
                os.makedirs(faces_dir, exist_ok=True)

                for file in os.listdir(faces_dir):
                    os.remove(os.path.join(faces_dir, file))

                for i, medoid_image in enumerate(medoid_images):
                    cv2.imwrite(
                        os.path.join(faces_dir, f"face_{i}.png"),
                        cvtColor(medoid_image, cv2.COLOR_RGB2BGR),
                    )
            except Exception as e:
                print(e)

        time.sleep(3)


# https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int) * -1

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new


def draw_box(frame, bounds, name):
    top, right, bottom, left = bounds
    if name is None:
        name = "Unknown"

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


# Function to process video frames and recognize faces
def frame_processing(
    video_capture,
    known_faces,
    frame_queue,
    face_queue,
    bounds,
    process_every_n_frames=3,
):
    global clustering
    frame_count = 0

    # Unzip the known_faces dictionary into two lists of names and encodings
    try:
        known_face_names, known_face_encodings = zip(*known_faces.items())
    except ValueError:
        known_face_names = []
        known_face_encodings = []

    while True:
        ret, frame = video_capture.read()  # Capture frame from the camera
        # OpenCV video_capture returns a boolean (True or False) value for us to confirm if videos actually being captured
        if not ret:
            break

        # Process only every nth frame to reduce load
        frame_count += 1
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

            bounds = []

            # Loop over each face found in the frame to see if it's someone we know
            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                cropped_face = image.crop((left, top, right, bottom))
                resized_face = cropped_face.resize((width, height))
                face = np.array(resized_face)
                # resized_face.show()
                # print(face)
                # print(face.shape)

                face_queue.put((face, face_encoding))

                index = None

                if clustering is not None:
                    index = dbscan_predict(clustering, [face_encoding])[0]

                bounds.append(((top, right, bottom, left), index))

                try:
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding, tolerance=0.6
                    )
                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                except Exception as e:
                    pass

            for bound, name in bounds:
                draw_box(frame, bound, str(name))

            # Put the frame and the recognized name (if any) into the queue
            frame_queue.put((frame, name))
        else:
            for bound, name in bounds:
                draw_box(frame, bound, str(name))

            # If not processing for recognition, still put the frame in the queue
            frame_queue.put((frame, None))
