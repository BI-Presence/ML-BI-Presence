from django.shortcuts import render
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response 
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from model_api.services.prediction import Prediction
import cv2
from django.http import StreamingHttpResponse, JsonResponse, HttpRequest
from django.core.files.uploadedfile import SimpleUploadedFile
import time
from collections import Counter

# Index
def index(request):
    return render(request, 'detection/index.html')

# Testing with Camera
def detect_faces_camera(request):
    # Initialize OpenCV Cascade Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Maximum number of predictions
    max_predictions = 10
    prediction_count = 0
    prediction_ids = []

    # Function to generate frames from camer
    def gen_frames():
        nonlocal prediction_count, prediction_ids
        cap = cv2.VideoCapture(0)
        while True:
            if prediction_count >= max_predictions:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Perform face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Pause the video feed and classify the detected face
                face_img = frame[y:y+h, x:x+w]
                _, jpeg = cv2.imencode('.jpg', face_img)
                face_bytes = jpeg.tobytes()
                result = classify_face(face_bytes)
                print(result)  # Log the classification result to the console

                # Collect prediction result and update count
                if 'UserID' in result:
                    prediction_ids.append(result['UserID'])
                    prediction_count += 1

            # Convert frame to JPEG format for web display
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Determine the most common UserID
        if prediction_ids:
            most_common_id = Counter(prediction_ids).most_common(1)[0][0]
            print(f"Most common UserID: {most_common_id}")
        else:
            print("No UserID detected.")

    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def classify_face(face_bytes):
    prediction_obj = Prediction()
    face_file = SimpleUploadedFile("detected_face.jpg", face_bytes, content_type="image/jpeg")
    mock_request = HttpRequest()
    mock_request.method = 'POST'
    mock_request.FILES['media'] = face_file
    response_dict = prediction_obj.predict(mock_request)
    return response_dict['response']
