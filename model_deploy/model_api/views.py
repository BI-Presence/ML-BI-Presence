from django.shortcuts import render
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response 
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from model_api.services.prediction import Prediction
from model_api.services.training import train_model
from model_api.services.training import check_new_uid
from model_api.services.live_predict import LivePrediction
import cv2
from django.http import JsonResponse, HttpRequest
from django.core.files.uploadedfile import SimpleUploadedFile
import time
from collections import Counter
import threading
from rest_framework.views import APIView
import base64
import numpy as np

# Index
def index(request):
    return render(request, 'detection/index.html')

# Function to classify face
def classify_face(face_bytes):
    prediction_obj = LivePrediction()
    face_file = SimpleUploadedFile("detected_face.jpg", face_bytes, content_type="image/jpeg")
    mock_request = HttpRequest()
    mock_request.method = 'POST'
    mock_request.FILES['media'] = face_file
    response_dict = prediction_obj.predict(mock_request)
    return response_dict['response']

# API endpoint for face detection and classification
class DetectFacesCameraView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @swagger_auto_schema(
        tags=['Live Detection Using Camera'],
        manual_parameters=[
            openapi.Parameter(
                name='image', 
                in_=openapi.IN_FORM, 
                type=openapi.TYPE_STRING, 
                description='Base64 encoded image',
                required=True
            )
        ],
        responses={
            200: openapi.Response(
                'Success',
                openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'UserID': openapi.Schema(type=openapi.TYPE_STRING, description='User ID of the detected face'),
                        'confidence': openapi.Schema(type=openapi.TYPE_NUMBER, format=openapi.FORMAT_FLOAT, description='Confidence level'),
                    }
                ),
                examples={
                    'application/json': {
                        "UserID": "18999687-c74f-419e-9377-8f4056b41612",
                        "confidence": 99.79705214500427,
                    }
                }
            ),
            400: 'Bad Request',
            500: 'Internal Server Error'
        }
    )  

    def post(self, request):
        
        image_base64 = request.data.get("image")
        if not image_base64:
            return JsonResponse({"error": "No image provided"}, status=400)

        # Decode the base64 image
        face_bytes = base64.b64decode(image_base64)

        # Initialize OpenCV Cascade Classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        nparr = np.fromstring(face_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform face detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return JsonResponse({"error": "No faces detected"}, status=200)

        # Take the first detected face for classification
        (x, y, w, h) = faces[0]
        face_img = img_np[y:y+h, x:x+w]
        _, jpeg = cv2.imencode('.jpg', face_img)
        face_bytes = jpeg.tobytes()

        result = classify_face(face_bytes)
        print (result)
        return JsonResponse(result, status=200)

class PredFacenetView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @swagger_auto_schema(
        tags=['Prediction'],
        manual_parameters=[
            openapi.Parameter(
                name='media', 
                in_=openapi.IN_FORM, 
                type=openapi.TYPE_FILE, 
                description='Image file',
                required=True
            )
        ],
        responses={
            200: openapi.Response(
                'Success', 
                openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING, description='Error status'),
                        'message': openapi.Schema(type=openapi.TYPE_STRING, description='Message'),
                        'predictionResult': openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            properties={
                                'UserID': openapi.Schema(type=openapi.TYPE_STRING, description='User ID'),
                                'confidence': openapi.Schema(type=openapi.TYPE_NUMBER, format=openapi.FORMAT_FLOAT, description='Confidence level'),
                            }
                        )
                    }
                ),
                examples={
                    'application/json': {
                        "error": "false",
                        "message": "success",
                        "predictionResult": {
                            "UserID": "18999687-c74f-419e-9377-8f4056b41612",
                            "confidence": 99.79705214500427,
                        }
                    }
                }
            ),
            400: 'Bad Request',
        }
    )
    def post(self, request):
        pred_obj = Prediction()
        response_dict = pred_obj.predict(request)
        response = response_dict['response']
        status_value = response_dict['status']
        return Response(response, status=status_value)

class TrainModelView(APIView):
    @swagger_auto_schema(
        tags=['Train model'],
        manual_parameters=[
            openapi.Parameter(
                name='train',
                in_=openapi.IN_QUERY,
                type=openapi.TYPE_BOOLEAN,
                description='Set to True to trigger model training.',
                required=True
            )
        ],
        responses={
            200: openapi.Response(
                'Model training started.',
                openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'message': openapi.Schema(type=openapi.TYPE_STRING, description='Message')
                    }
                ),
                examples={
                    'application/json': {
                        "message": "Pelatihan model dimulai."
                    }
                }
            ),
            400: openapi.Response(
                'Bad Request',
                openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING, description='Error status'),
                        'message': openapi.Schema(type=openapi.TYPE_STRING, description='Message')
                    }
                ),
                examples={
                    'application/json': {
                        "error": "true",
                        "message": 'Setel parameter train ke true untuk memulai pelatihan.'
                    }
                }
            ),
            500: openapi.Response(
                'Internal Server Error',
                openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'error': openapi.Schema(type=openapi.TYPE_STRING, description='Error status'),
                        'message': openapi.Schema(type=openapi.TYPE_STRING, description='Message')
                    }
                ),
                examples={
                    'application/json': {
                        "error": "true",
                        "message": "Error message details"
                    }
                }
            )
        }
    )
    def post(self, request):
        train_param = request.query_params.get('train', '').lower()

        if train_param != 'true':
            return Response({'error': 'true', 'message': 'Setel parameter train ke true untuk memulai pelatihan.'}, status=400)

        try:
            new_uid = check_new_uid()
            if new_uid is None:
                return Response({'error': 'true', 'message': 'Tidak ada UID baru. Pelatihan tidak dimulai.'}, status=500)
            thread = threading.Thread(target=train_model)
            thread.start()
            return Response({'error': 'false', 'message': 'Pelatihan model dimulai.'}, status=200)

        except Exception as e:
            return Response({'error': 'true', 'message': str(e)}, status=500)

