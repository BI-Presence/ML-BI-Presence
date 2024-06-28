from django.db import models

# Create your models here.
class SaveImagesModel(models.Model):
    fileName = models.ImageField(upload_to='saved_images/')  # Define the field to store the image
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Timestamp of when the image was uploaded
