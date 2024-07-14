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
MODEL_H5_PATH = os.path.normpath(BASE_PATH + os.sep + 'model_h5'+ os.sep + 'updated_mtcnn_facenet_mlp_model.h5')
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

# Pre-warm the models with a dummy image
def warm_up_models():
    # Create a dummy image
    dummy_img = np.zeros((160, 160, 3), dtype=np.float32)
    dummy_img_expanded = np.expand_dims(dummy_img, axis=0)
    
    # Warm-up FaceNet
    EMBEDDER.embeddings(dummy_img_expanded)
    print("FaceNet model warmed up.")
    
    # Warm-up MTCNN
    dummy_img_uint8 = (dummy_img * 255).astype(np.uint8)  # Convert to uint8
    IMG_DETECTOR.detect_faces(dummy_img_uint8)
    print("MTCNN model warmed up.")

# Warm-up the models in a separate thread
threading.Thread(target=warm_up_models).start()

# Get prediction result
def get_prediction(embedding):
    # Convert the embedding to the correct shape (2D array)
    embedding = np.expand_dims(embedding, axis=0)
    
    # Make Prediction
    predict_proba = MODEL.predict(embedding)[0]
    predicted_class = np.argmax(predict_proba)
    predicted_label = encoder.inverse_transform([predicted_class])[0]
    confidence_score = predict_proba[predicted_class]

    # Check confidence score and determine if the prediction should be considered unknown
    if confidence_score < 0.9:
        predicted_label = "unknown"

    # Convert confidence score to percentage
    confidence_percentage = confidence_score * 100

    return predicted_label, confidence_percentage

class Prediction:
    # Set the parser classes to handle multipart file uploads
    parser_classes = [MultiPartParser]

    def preprocess_image(self, image_file, new_size=(480, 480)):
        # file_extension = os.path.splitext(image_file.name)[1].lower()
        # if file_extension not in ['.jpg', '.jpeg', '.png']:
        #     raise Exception("Unsupported file type.")

        # Convert the uploaded image file to a NumPy array
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)

        # Decode the image using OpenCV
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

        # Convert the image to RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Resize the image to the new size while maintaining aspect ratio
        height, width, _ = img.shape
        scale = min(new_size[0] / width, new_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv.resize(img, (new_width, new_height))

        # Detect faces in the resized image
        detections = IMG_DETECTOR.detect_faces(resized_img)
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
        face_img = img[y:y+h, x:x+w]

        # Resize the cropped face image to 160x160
        face_img = cv.resize(face_img, (160, 160))

        # Get the embedding (feature vector) for the resized face image using FaceNet
        face_img = face_img.astype('float32')  # 3D (160x160x3)
        face_img = np.expand_dims(face_img, axis=0)  # 4D (1x160x160x3)
        embedding = EMBEDDER.embeddings(face_img)[0]  # 512D image (1x512)

        return embedding

    # def save_image_to_database(self, image_file):
    #     # Save the image to the database
    #     try:
    #         # Create an instance of ImageModel and save the image
    #         new_image = SaveImagesModel(fileName=image_file)
    #         new_image.save()
            
    #         # Generate the URL for the saved image
    #         image_url = f'/media/{new_image.fileName}'
            
    #         return new_image, image_url  # Optionally return the saved instance and URL for further processing
    #     except Exception as e:
    #         raise Exception(f"Failed to save image to database: {str(e)}")

    def predict(self, request):
        # Initialize the return dictionary
        return_dict = {}
        try:
            if 'media' not in request.FILES:
                raise Exception("No media file found in the request.")
            
            # Load the image file from the request
            image_file = request.FILES['media']
            
            # Preprocess the image
            embedding = self.preprocess_image(image_file)

            # Make Prediction
            predicted_label, confidence_score = get_prediction(embedding)

            # # Save the image to the database and get the URL
            # saved_image, image_url = self.save_image_to_database(image_file)

            # Create prediction result dictionary
            prediction_result = {
                "UserID": predicted_label,
                "confidence": confidence_score,  # Ensure confidence_score is in percentage
                # "imageURL": image_url
            }

            # Create result dictionary
            result = {
                "error": "false",
                "message": "success",
                "predictionResult": prediction_result
            }

            # Assign result to response key in return_dict
            return_dict['response'] = result
            return_dict['status'] = status.HTTP_200_OK

        except Exception as e:
            # Handle exceptions and prepare error response
            result = {
                "error": "true",
                "message": str(e)
            }
            return_dict['response'] = result
            return_dict['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR

        return return_dict