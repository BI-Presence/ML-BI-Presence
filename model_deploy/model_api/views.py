import cv2
import time
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse, HttpRequest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response 
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.views import APIView
from model_api.services.prediction import Prediction
from model_api.services.training import train_model, check_new_uid
from model_api.services.live_predict import LivePrediction
import threading
import requests

# Index
def index(request):
    return render(request, 'detection/index.html')

# Testing with Camera
def detect_faces_camera(request):
    # Initialize OpenCV Cascade Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    blue_color = (153, 86, 1)

    # Function to generate frames from camera
    def gen_frames():
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Draw rectangles around detected faces and display UserID and confidence
            for (x, y, w, h) in faces:
              
                # Pause the video feed and classify the detected face
                face_img = frame[y:y+h, x:x+w]
                _, jpeg = cv2.imencode('.jpg', face_img)
                face_bytes = jpeg.tobytes()
                result = classify_face(face_bytes)

                user_id = result['predictionResult']['UserID']
                confidence = result['predictionResult']['confidence']

                if confidence >= 50:

                    cv2.rectangle(frame, (x, y), (x+w, y+h), blue_color, 2)

                    if user_id == "unknown":
                        user_id = "00000000-0000-0000-0000-000000000000"

                    response = send_api_request(user_id, confidence)
                    
                    # Display fullName and confidence inside the blue rectangle
                    if response:
                        text = f'Name: {response["fullName"]}, Conf: {confidence:.2f}'
                    else: 
                        text = 'Tidak ada response dari API'

                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_w, text_h = text_size

                    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), blue_color, cv2.FILLED)
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Convert frame to JPEG format for web display
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    response = StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
    return response

def send_api_request(user_id, confidence):
    url = "https://97cf-103-243-178-32.ngrok-free.app/api/presences/ml-result" # URL endpoint
    print('SEND API :',user_id, confidence)

    # Create a dictionary with user_id and confidence
    data = {
        "confidence": confidence,
        "userId": user_id
    }

    print('SEND API :',data)

    try:
        response = requests.post(url, json=data)
        print(f"API request status code: {response.status_code}")

        if response.status_code == 200:
            # Get JSON response data
            json_response = response.json()
            print(f"JSON Response:{json_response}")
            return json_response
        else:
            print(f"API request failed with status code: {response.status_code}")
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")

    return None

def classify_face(request):
    if request.method == 'POST' and request.FILES.get('media'):
        face_file = request.FILES['media']
        prediction_obj = LivePrediction()
        mock_request = HttpRequest()
        mock_request.method = 'POST'
        mock_request.FILES['media'] = face_file
        response_dict = prediction_obj.predict(mock_request)
        print(response_dict['response'])
        return JsonResponse({'response': response_dict['response']})
    return JsonResponse({'error': 'Invalid request'}, status=400)

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
                                'timestamp': openapi.Schema(type=openapi.TYPE_STRING, description='Timestamp'),
                                'confidence': openapi.Schema(type=openapi.TYPE_NUMBER, format=openapi.FORMAT_FLOAT, description='Confidence level'),
                                'imageID': openapi.Schema(type=openapi.TYPE_INTEGER, description='Image ID'),
                            }
                        )
                    }
                ),
                examples={
                    'application/json': {
                        "error": "false",
                        "message": "success",
                        "predictionResult": {
                            "UserID": "vicky",
                            "timestamp": "2024-07-03T14:05:49.439126",
                            "confidence": 0.9121062518765518,
                            "imageID": 13
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
                        "message": "Model training started."
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
                        "message": 'Set train parameter to true to start training.'
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
            return Response({'error': 'true', 'message': 'Set the train parameter to true to start training.'}, status=400)

        try:
            new_uid = check_new_uid()
            if new_uid is None:
                return Response({'error': 'true', 'message': 'There is no new UID. Training not started.'}, status=500)
            thread = threading.Thread(target=train_model)
            thread.start()
            return Response({'error': 'false', 'message': 'Model training started.'}, status=200)
 
        except Exception as e:
            return Response({'error': 'true', 'message': str(e)}, status=500)