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
import subprocess

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
        t_img = cv.imread(filename)
        t_img = cv.cvtColor(t_img, cv.COLOR_BGR2RGB)

        # Resize the image to the new size while maintaining aspect ratio
        height, width, _ = t_img.shape
        scale = min(new_size[0] / width, new_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv.resize(t_img, (new_width, new_height))

        detections = self.detector.detect_faces(resized_img)
        if not detections:
            raise Exception("No faces detected in the image.")

        x, y, w, h = detections[0]['box']

        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        face_img = t_img[y:y+h, x:x+w]
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

        faceloading = FACELOADING(DATASET_PATH)
        X, new_Y = faceloading.load_classes()

        embedder = FaceNet()

        def get_embedding(face_img):
            face_img = face_img.astype('float32')  # 3D(160x160x3)
            face_img = np.expand_dims(face_img, axis=0)
            yhat = embedder.embeddings(face_img)
            return yhat[0]  # 512D image (1x1x512)

        for img in X:
            EMBEDDED_X.append(get_embedding(img))

        Y.extend(new_Y)
        unique_labels = []
        seen_labels = set()

        for label in Y:
            if label not in seen_labels:
                unique_labels.append(label)
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

        encoder = LabelEncoder()
        encoder.fit(Y)
        Y_encoded = encoder.transform(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y_encoded, test_size=0.2, shuffle=True, random_state=17)
        in_encoder = Normalizer(norm='l2')

        # Apply normalization 
        X_train_norm = in_encoder.transform(X_train)
        X_test_norm = in_encoder.transform(X_test)

        # Convert labels to one-hot encoding
        num_classes = len(np.unique(Y_encoded))
        Y_train_categorical = to_categorical(Y_train, num_classes)
        Y_test_categorical = to_categorical(Y_test, num_classes)

        # Define the ANN model
        model = Sequential()
        model.add(Dense(512, input_shape=(X_train_norm.shape[1],), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # Compile and train the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train_norm, Y_train_categorical, epochs=100, batch_size=32, validation_data=(X_test_norm, Y_test_categorical))

        # Evaluate the model 
        train_loss = history.history['loss'][-1]
        train_accuracy = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]

        # Print result training and validation
        print(f'Train loss: {train_loss} / Train accuracy: {train_accuracy}')
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

        # Run the batch file to restart the Django server
        restart_server_file = os.path.normpath(CONFIG_PATH + os.sep + 'restart_server.bat')
        subprocess.run([restart_server_file], check=True)

    except Exception as e:
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
    url = "http://localhost:5124/api/trainings/update-trainings"
    data = []

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