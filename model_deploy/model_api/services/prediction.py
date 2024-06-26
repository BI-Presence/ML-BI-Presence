import os
import numpy as np
import pickle
import cv2 as cv
from rest_framework import status
from sklearn.preprocessing import LabelEncoder
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from datetime import datetime
from rest_framework.parsers import MultiPartParser

# Set TF_CPP_MIN_LOG_LEVEL to suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define directory path
base_path = os.getcwd()
pickle_path = os.path.normpath(base_path + os.sep + 'pickle')
log_path = os.path.normpath(base_path + os.sep + 'log')

# Load Saved Model
pickle_file = os.path.normpath(pickle_path + os.sep + 'fix_model_facenet_160x160.pkl')
with open(pickle_file, 'rb') as f:
    fix_model_facenet = pickle.load(f)

# Initialize MTCNN and FaceNet models
img_detector = MTCNN()
embedder = FaceNet()

# Load label encoder (when adding a new class or more classes, it is required to edit the label as well)
encoder = LabelEncoder()
LABEL = ['abed', 'budi', 'gibran', 'iyal', 'vicky']
encoder.fit(LABEL)

def get_prediction(embedding):
    # Perform prediction
    predict = fix_model_facenet.predict([embedding])
    predict_proba = fix_model_facenet.predict_proba([embedding])
    return predict, predict_proba

class Prediction:
    # Set the parser classes to handle multipart file uploads
    parser_classes = [MultiPartParser]

    def preprocess_image(self, image_file, new_size=(480, 480)):
        file_extension = os.path.splitext(image_file.name)[1].lower()
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            raise Exception("Unsupported file type.")

        # Convert the uploaded image file to a NumPy array
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        t_img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

        # Convert the image to RGB
        t_img = cv.cvtColor(t_img, cv.COLOR_BGR2RGB)

        # Resize the image to the new size while maintaining aspect ratio
        height, width, _ = t_img.shape
        scale = min(new_size[0] / width, new_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv.resize(t_img, (new_width, new_height))

        # Detect faces in the resized image
        detections = img_detector.detect_faces(resized_img)
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
        face_img = cv.resize(face_img, (160, 160))

        # Get the embedding (feature vector) for the resized face image using FaceNet
        face_img = face_img.astype('float32')  # 3D (160x160x3)
        face_img = np.expand_dims(face_img, axis=0)  # 4D (1x160x160x3)
        test_embedding = embedder.embeddings(face_img)[0]  # 512D image (1x512)

        return test_embedding

    def predict(self, request):
        # Initialize the return dictionary
        return_dict = {}
        try:
            if 'media' not in request.FILES:
                raise Exception("No media file found in the request.")
            
            # Load the image file from the request
            image_file = request.FILES['media']
            
            # Preprocess the image
            test_embedding = self.preprocess_image(image_file)

            # Make Prediction
            predict, predict_proba = get_prediction(test_embedding)

            # Decode the prediction
            predicted_label = encoder.inverse_transform(predict)[0]
            confidence_score = predict_proba[0][predict[0]]

            # Check confidence score and determine if the prediction should be considered unknown
            if confidence_score < 0.5:
                predicted_label = "unknown"

            timestamp = datetime.now().isoformat()

            # Create prediction result dictionary
            prediction_result = {
                "UserID": predicted_label,
                "timestamp": timestamp,
                "confidence": float(confidence_score)  # Ensure confidence_score is a float
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
