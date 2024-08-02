import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2 as cv
import tensorflow as tf
import tensorflow_hub as hub
from rest_framework import status
from sklearn.preprocessing import LabelEncoder
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from rest_framework.parsers import MultiPartParser
import threading
from model_api.models import SaveImagesModel

# Define directory path
BASE_PATH = os.getcwd()
MODEL_H5_PATH = os.path.normpath(BASE_PATH + os.sep + 'model_h5'+ os.sep + 'updated_mtcnn_facenet_ann_model.h5')
MODEL = tf.keras.models.load_model(MODEL_H5_PATH, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
CONFIG_PATH = os.path.normpath(BASE_PATH + os.sep + 'config')

# Initialize MTCNN and FaceNet models
IMG_DETECTOR = MTCNN()
EMBEDDER = FaceNet()

# Load labels from a text file (when adding a new class or more classes, it is required to edit the file as well)
def load_labels(LABEL_FILE_PATH):
    with open(LABEL_FILE_PATH, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

# Load label encoder
encoder = LabelEncoder()
label_file = os.path.normpath(CONFIG_PATH + os.sep + 'labels.txt')
LABEL = load_labels(label_file)
encoder.fit(LABEL)

# Pre-warm the models
def warm_up_models():
    dummy_img = np.zeros((160, 160, 3), dtype=np.float32)
    dummy_img_expanded = np.expand_dims(dummy_img, axis=0)
    
    EMBEDDER.embeddings(dummy_img_expanded)
    print("FaceNet model warmed up.")
    
    dummy_img_uint8 = (dummy_img * 255).astype(np.uint8)
    IMG_DETECTOR.detect_faces(dummy_img_uint8)
    print("MTCNN model warmed up.")

threading.Thread(target=warm_up_models).start()

def get_prediction(embedding):
    embedding = np.expand_dims(embedding, axis=0)
    predict_proba = MODEL.predict(embedding)[0]
    predicted_class = np.argmax(predict_proba)
    predicted_label = encoder.inverse_transform([predicted_class])[0]
    confidence_score = predict_proba[predicted_class]

    # Check confidence score and determine if the prediction should be considered unknown
    if confidence_score < 0.9:
        predicted_label = "unknown"

    confidence_percentage = confidence_score * 100
    return predicted_label, confidence_percentage

class Prediction:
    parser_classes = [MultiPartParser]

    def preprocess_image(self, image_file, new_size=(480, 480)):

        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        height, width, _ = img.shape
        scale = min(new_size[0] / width, new_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv.resize(img, (new_width, new_height))

        detections = IMG_DETECTOR.detect_faces(resized_img)
        if not detections:
            raise Exception("No faces detected in the image.")

        x, y, w, h = detections[0]['box']

        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        face_img = img[y:y+h, x:x+w]
        face_img = cv.resize(face_img, (160, 160))

        # Get the embedding (feature vector) for the resized face image using FaceNet
        face_img = face_img.astype('float32')  # 3D (160x160x3)
        face_img = np.expand_dims(face_img, axis=0)  # 4D (1x160x160x3)
        embedding = EMBEDDER.embeddings(face_img)[0]  # 512D image (1x512)

        return embedding

    def predict(self, request):
        return_dict = {}
        try:
            if 'media' not in request.FILES:
                raise Exception("No media file found in the request.")
            
            image_file = request.FILES['media']
            embedding = self.preprocess_image(image_file)
            predicted_label, confidence_score = get_prediction(embedding)

            # Create prediction result
            prediction_result = {
                "UserID": predicted_label,
                "confidence": confidence_score
            }

            # Create result dictionary
            result = {
                "error": "false",
                "message": "success",
                "predictionResult": prediction_result
            }

            return_dict['response'] = result
            return_dict['status'] = status.HTTP_200_OK

        except Exception as e:
            result = {
                "error": "true",
                "message": str(e)
            }
            return_dict['response'] = result
            return_dict['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR

        return return_dict