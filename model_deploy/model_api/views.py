from django.shortcuts import render
from rest_framework.response import Response 
from rest_framework.views import APIView
from model_api.services.prediction import Prediction

# Create your views here.
class PredFacenetView(APIView): 
    def post(self,request):
        pred_obj=Prediction()
        response_dict=pred_obj.predict(request)
        response=response_dict['response']
        status_value=response_dict['status']
        return Response(response,status_value)