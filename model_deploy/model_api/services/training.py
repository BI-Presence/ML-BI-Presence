import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential
from keras.utils import to_categorical
from keras_facenet import FaceNet
import cv2 as cv
from mtcnn.mtcnn import MTCNN
import shutil
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BASE_PATH = os.getcwd()
CONFIG_PATH = os.path.normpath(BASE_PATH + os.sep + 'config')
DATASET_PATH = os.path.normpath(BASE_PATH + os.sep + 'dataset')
MODEL_H5_PATH = os.path.normpath(BASE_PATH + os.sep + 'model_h5' + os.sep + 'updated_mtcnn_facenet_ann_model.h5')

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def preprocess_image(self, filename, new_size=(480, 480)):
        # Load the image
        t_img = cv.imread(filename)

        # Convert the image to RGB
        t_img = cv.cvtColor(t_img, cv.COLOR_BGR2RGB)

        # Resize the image to the new size while maintaining aspect ratio
        height, width, _ = t_img.shape
        scale = min(new_size[0] / width, new_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv.resize(t_img, (new_width, new_height))

        # Detect faces in the resized image
        detections = self.detector.detect_faces(resized_img)
        if not detections:
            raise Exception("No faces detected in the image.")

        # Extract the coordinates and size of the bounding box from the first detection result
        x, y, w, h = detections[0]['box']

        # Scale the coordinates back to the original image size
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        # Crop the detected face region from the original image using the bounding box coordinates
        face_img = t_img[y:y+h, x:x+w]

        # Resize the cropped face image to 160x160
        face_img = cv.resize(face_img, self.target_size)

        return face_img

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                single_face = self.preprocess_image(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)

def check_new_uid():
    try:
        NEW_UID = os.listdir(DATASET_PATH)
        if not NEW_UID:
            return None
        NEW_UID.sort()
        return np.array(NEW_UID)
    except Exception as e:
        print(f"Error accessing or listing folders in DATASET_PATH: {e}")
        return None

def train_model():

    TRAIN_ON_PROCESS = 1
    TRAIN_SUCESS = 2
    TRAIN_FAIL = -1
    
    # Get list of folders (classes) in DATASET_PATH
    new_uid = check_new_uid()
    print (new_uid)

    # set status to on proses
    train_status = TRAIN_ON_PROCESS
    send_api_request(new_uid, train_status)

    try:
        # Load faces_embedding (feature vector)
        try:
            embeddings_file = os.path.normpath(CONFIG_PATH + os.sep + 'updated_faces_embeddings_done.npz')
            data = np.load(embeddings_file)
            EMBEDDED_X = list(data['arr_0'])
            Y = list(data['arr_1'])
        except FileNotFoundError:
            EMBEDDED_X = []
            Y = []

        # Print the number of elements in Y
        print(f"Amount of data: {len(Y)}")

        # Initiate the FACELOADING class with the dataset directory path
        faceloading = FACELOADING(DATASET_PATH)

        # Load the face images (X) and their corresponding labels (Y) from the dataset
        X, new_Y = faceloading.load_classes()

        # Instantiate the FaceNet model
        embedder = FaceNet()

        # Define a function to get the embedding of a face image
        def get_embedding(face_img):
            face_img = face_img.astype('float32')  # 3D(160x160x3)
            face_img = np.expand_dims(face_img, axis=0)
            yhat = embedder.embeddings(face_img)
            return yhat[0]  # 512D image (1x1x512)

        # Loop through each image in the dataset X to get the embedding for the current image and append it to the list
        for img in X:
            EMBEDDED_X.append(get_embedding(img))

        # Append new labels to existing labels
        Y.extend(new_Y)

        # Initialize an empty list to store unique labels in order
        unique_labels = []

        # Set to track seen labels
        seen_labels = set()

        # Iterate through each label in Y
        for label in Y:
            # Check if the label has not been seen before
            if label not in seen_labels:
                # Add the label to the list of unique labels
                unique_labels.append(label)
                # Add the label to the set of seen labels
                seen_labels.add(label)

        # Sort the unique labels alphabetically
        unique_labels.sort()

        # Write the unique labels to a text file
        label_file = os.path.normpath(CONFIG_PATH + os.sep + 'labels.txt')
        with open(label_file, 'w') as file:
            for label in unique_labels:
                file.write(f"{label}\n")

        print(f"Labels: {unique_labels}")
        print("Labels have been saved to 'labels.txt' file.")

        # Save the updated embeddings and labels into a compressed NumPy archive file (.npz)
        embeddings_save_file = embeddings_file  # embeddings file path
        np.savez_compressed(embeddings_save_file, EMBEDDED_X, Y)

        # Initiate a LabelEncoder object
        encoder = LabelEncoder()

        # Fit the LabelEncoder to the array Y to learn the classes
        encoder.fit(Y)

        # Transform the labels in Y to encoded labels
        Y_encoded = encoder.transform(Y)

        # Split the data into training (80%) and testing (20%)
        X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y_encoded, test_size=0.2, shuffle=True, random_state=17)

        # Initiate normalizer
        in_encoder = Normalizer(norm='l2')

        # Apply normalization to training data
        X_train_norm = in_encoder.transform(X_train)

        # Apply normalization to testing data
        X_test_norm = in_encoder.transform(X_test)

        # Convert labels to one-hot encoding
        num_classes = len(np.unique(Y_encoded))
        Y_train_categorical = to_categorical(Y_train, num_classes)
        Y_test_categorical = to_categorical(Y_test, num_classes)

        print(num_classes)

        # Define the ANN model
        model = Sequential()
        model.add(Dense(512, input_shape=(X_train_norm.shape[1],), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train_norm, Y_train_categorical, epochs=100, batch_size=32, validation_data=(X_test_norm, Y_test_categorical))

        # Evaluate the model 
        train_loss = history.history['loss'][-1]
        train_accuracy = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]

        # Print training loss and accuracy
        print(f'Train loss: {train_loss} / Train accuracy: {train_accuracy}')

        # Print validation (test) loss and accuracy
        print(f'Validation loss: {val_loss} / Validation accuracy: {val_accuracy}')

        # Save the model
        model_save_file = MODEL_H5_PATH
        model.save(model_save_file)
        print(f"Model saved to {model_save_file}")

        # Clear the dataset folder
        clear_dataset_folder()
        print(f"Dataset folder cleared")

        # Send API request with UID and status
        train_status = TRAIN_SUCESS
        send_api_request(new_uid, train_status)

    except Exception as e:
        # Send status as -1 if an error occurs
        train_status = TRAIN_FAIL
        send_api_request(new_uid, train_status)
        print(f"An error occurred: {e}")
        return


def clear_dataset_folder():
    # Remove all files and directories in the dataset folder
    for filename in os.listdir(DATASET_PATH):
        file_path = os.path.join(DATASET_PATH, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def send_api_request(uid_list, status):
    url = "https://dbb4-103-243-178-32.ngrok-free.app/api/trainings/update-trainings"
    data = []

    # Loop through each UID in uid_list and create a dictionary for each UID and status
    for uid in uid_list:
        data.append({
            "userId": uid,
            "status": status
        })

    try:
        response = requests.patch(url, json=data)
        response.raise_for_status()
        print(f"API request successful: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")

if __name__ == '__main__':
    check_new_uid()
    train_model()