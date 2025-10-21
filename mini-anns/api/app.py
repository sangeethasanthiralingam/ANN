"""
Flask API for Mini-ANNs Project
RESTful endpoints for model inference
"""

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
from PIL import Image
import io
import sys
import os

# Add scripts directory to path
sys.path.append('../scripts')

app = Flask(__name__)

# Define model architectures
class TinyImageClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(TinyImageClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MiniCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mnist_model = TinyImageClassifier().to(device)
cifar_model = MiniCNN().to(device)

# Class names
mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def home():
    """Home endpoint with API information."""
    return jsonify({
        "message": "Mini-ANNs API Server",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/predict/mnist": "MNIST digit classification",
            "/predict/cifar": "CIFAR-10 object classification",
            "/health": "Health check",
            "/models": "Available models"
        },
        "usage": {
            "mnist": "POST /predict/mnist with base64 image",
            "cifar": "POST /predict/cifar with base64 image"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "models_loaded": True
    })

@app.route('/models')
def models():
    """List available models."""
    return jsonify({
        "models": {
            "mnist": {
                "name": "Tiny Image Classifier",
                "input_size": "28x28 grayscale",
                "output_classes": 10,
                "classes": mnist_classes
            },
            "cifar": {
                "name": "Mini CNN",
                "input_size": "32x32 RGB",
                "output_classes": 10,
                "classes": cifar_classes
            }
        }
    })

@app.route('/predict/mnist', methods=['POST'])
def predict_mnist():
    """Predict MNIST digit from base64 image."""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image = np.array(image) / 255.0  # Normalize
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
        
        # Predict
        mnist_model.eval()
        with torch.no_grad():
            output = mnist_model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return jsonify({
            "prediction": mnist_classes[predicted_class],
            "confidence": confidence,
            "probabilities": {
                mnist_classes[i]: float(probabilities[0][i]) 
                for i in range(len(mnist_classes))
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/cifar', methods=['POST'])
def predict_cifar():
    """Predict CIFAR-10 class from base64 image."""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        image = image.convert('RGB')  # Convert to RGB
        image = image.resize((32, 32))  # Resize to 32x32
        image = np.array(image) / 255.0  # Normalize
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Predict
        cifar_model.eval()
        with torch.no_grad():
            output = cifar_model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return jsonify({
            "prediction": cifar_classes[predicted_class],
            "confidence": confidence,
            "probabilities": {
                cifar_classes[i]: float(probabilities[0][i]) 
                for i in range(len(cifar_classes))
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/synthetic', methods=['POST'])
def predict_synthetic():
    """Generate synthetic data and make predictions."""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'mnist')
        num_samples = data.get('num_samples', 1)
        
        if model_type == 'mnist':
            # Generate random MNIST-like data
            synthetic_data = torch.randn(num_samples, 1, 28, 28).to(device)
            model = mnist_model
            classes = mnist_classes
        else:
            # Generate random CIFAR-like data
            synthetic_data = torch.randn(num_samples, 3, 32, 32).to(device)
            model = cifar_model
            classes = cifar_classes
        
        # Predict
        model.eval()
        with torch.no_grad():
            output = model(synthetic_data)
            probabilities = F.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
        
        results = []
        for i in range(num_samples):
            pred_class = predictions[i].item()
            confidence = probabilities[i][pred_class].item()
            results.append({
                "sample": i,
                "prediction": classes[pred_class],
                "confidence": confidence
            })
        
        return jsonify({
            "model_type": model_type,
            "num_samples": num_samples,
            "predictions": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Mini-ANNs API Server...")
    print(f"📱 Device: {device}")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📚 API Documentation: http://localhost:5000")
    print("🔍 Health Check: http://localhost:5000/health")
    print("📋 Available Models: http://localhost:5000/models")
    print("\n" + "="*50)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )