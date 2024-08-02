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

def index(request):
    return render(request, 'detection/index.html')

def send_api_request(user_id, confidence):

    url = "http://localhost:5124/api/presences/ml-result"

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
        
        if 'response' in response_dict:
            user_id = response_dict['response']['predictionResult']['UserID']
            confidence = response_dict['response']['predictionResult']['confidence']
            response = send_api_request(user_id, confidence)

            return JsonResponse({
                'confidence': confidence,
                'response': response
            })
        else:
            return JsonResponse({'error': 'Invalid response format'}, status=500)
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
        try:
            new_uid = check_new_uid()
            if new_uid is None:
                return Response({'error': 'true', 'message': 'There is no new UID. Training not started.'}, status=500)
            thread = threading.Thread(target=train_model)
            thread.start()
            return Response({'error': 'false', 'message': 'Model training started.'}, status=200)
 
        except Exception as e:
            return Response({'error': 'true', 'message': str(e)}, status=500)