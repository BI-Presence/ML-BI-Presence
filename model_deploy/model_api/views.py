from django.shortcuts import render
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response 
from rest_framework.views import APIView
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from model_api.services.prediction import Prediction
import cv2
from django.shortcuts import render
from django.http import StreamingHttpResponse

# Index
def index(request):
    return render(request, 'detection/index.html')

# Testing with Camera
def detect_faces_camera(request):
    # Initialize OpenCV Cascade Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Convert frame to JPEG format for web display
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# Testing with Camera

class PredFacenetView(APIView): 
    parser_classes = (MultiPartParser, FormParser)

    @swagger_auto_schema(
        # operation_description="Upload an image file to get a prediction.",
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
