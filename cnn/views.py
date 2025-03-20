from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.template.defaulttags import register
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights  # Import the weights class
from PIL import Image
import numpy as np
import os

# Custom template filter to split strings
@register.filter(name='split')
def split(value, key):
    """
    Returns the value split by key.
    """
    return value.split(key)

# Define the ResNet18 model
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Use IMAGENET1K_V1 weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust output layer for our task

    def forward(self, x):
        return self.model(x)

# Global model initialization (to avoid reloading on every request)
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Load pre-trained weights using IMAGENET1K_V1
model.fc = nn.Linear(model.fc.in_features, 28)  # Adjust output layer for your task

# Load the model weights on the CPU
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fruit_vegetable_model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
model.eval()  # Set the model to evaluation mode

# Class labels
CLASS_LABELS = {
    0: "Apple_Healthy",
    1: "Apple_Rotten",
    2: "Banana_Healthy",
    3: "Banana_Rotten", 
    4: "Bellpepper_Healthy",
    5: "Bellpepper_Rotten",
    6: "Carrot_Healthy",
    7: "Carrot_Rotten",
    8: "Cucumber_Healthy",
    9: "Cucumber_Rotten",
    10: "Grape_Healthy",
    11: "Grape_Rotten",
    12: "Guava_Healthy",
    13: "Guava_Rotten",
    14: "Jujube_Healthy",
    15: "Jujube_Rotten",
    16: "Mango_Healthy",
    17: "Mango_Rotten",
    18: "Orange_Healthy",
    19: "Orange_Rotten",
    20: "Pomegranate_Healthy",
    21: "Pomegranate_Rotten",
    22: "Potato_Healthy",
    23: "Potato_Rotten",
    24: "Strawberry_Healthy",
    25: "Strawberry_Rotten",
    26: "Tomato_Healthy",
    27: "Tomato_Rotten"
}

# Define transformation for image preprocessing (Resize, Normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def index(request):
    prediction_result = None
    error = None
    
    # If the request is a POST request, handle the uploaded file
    if request.method == 'POST':
        if request.FILES.get('uploaded_file'):
            try:
                uploaded_file = request.FILES['uploaded_file']
                fs = FileSystemStorage(location='/tmp')  # Use a temporary directory
                filename = fs.save(uploaded_file.name, uploaded_file)
                file_path = fs.path(filename)
                
                # Process the image and get prediction
                prediction_result = process_image(file_path, uploaded_file.name, fs)
                
            except Exception as e:
                error = f"Error processing image: {str(e)}"
        else:
            error = "No image file was uploaded."
    
    return render(request, 'cnn/index.html', {
        'result': prediction_result,
        'error': error
    })

def process_image(file_path, original_filename, fs):
    try:
        # Open the image file
        img = Image.open(file_path)
        img = img.convert('RGB')  # Ensure 3 channels (RGB)
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Move the tensor to the CPU
        device = torch.device("cpu")
        img_tensor = img_tensor.to(device)
        model.to(device)

        # Make predictions using the model
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_label = CLASS_LABELS[predicted_class.item()]
            confidence_percentage = round(confidence.item() * 100, 2)

        # Create result dictionary
        result = {
            'filename': original_filename,
            'prediction': predicted_label,
            'confidence': confidence_percentage
        }

        # Clean up - delete the temporary file
        fs.delete(os.path.basename(file_path))
        
        return result
    
    except Exception as e:
        # Clean up in case of error
        try:
            fs.delete(os.path.basename(file_path))
        except:
            pass
        raise e
