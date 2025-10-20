"""
Flask API for Mini-ANNs Project
RESTful API endpoints for model inference and training
"""

from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import io
import base64
from PIL import Image
import sys

# Add parent directory to path
sys.path.append('..')
from scripts.utils import get_device, set_seed

# Initialize Flask app
app = Flask(__name__)

# Set device
device = get_device()
set_seed(42)

# Model definitions
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

class MiniAutoencoder(nn.Module):
    def __init__(self, input_size=784, hidden_size=32):
        super(MiniAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Global model instances
models = {
    'classifier': TinyImageClassifier().to(device),
    'autoencoder': MiniAutoencoder().to(device),
    'timeseries': TimeSeriesLSTM().to(device)
}

# CIFAR-10 class names
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Mini-ANNs API',
        'version': '1.0.0',
        'endpoints': {
            '/predict/classifier': 'POST - Image classification',
            '/predict/autoencoder': 'POST - Image reconstruction',
            '/predict/timeseries': 'POST - Time series forecasting',
            '/models': 'GET - List available models',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'models_loaded': list(models.keys())
    })

@app.route('/models')
def list_models():
    """List available models"""
    model_info = {}
    for name, model in models.items():
        model_info[name] = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'size_mb': sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024,
            'device': str(device)
        }
    return jsonify(model_info)

@app.route('/predict/classifier', methods=['POST'])
def predict_classifier():
    """Image classification endpoint"""
    try:
        # Get image data
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0).to(device)
        
        # Make prediction
        model = models['classifier']
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
        
        # Return results
        return jsonify({
            'predicted_class': int(predicted_class),
            'confidence': float(probabilities[0, predicted_class]),
            'probabilities': probabilities[0].cpu().numpy().tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/autoencoder', methods=['POST'])
def predict_autoencoder():
    """Image reconstruction endpoint"""
    try:
        # Get image data
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image) / 255.0
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0).to(device)
        
        # Make prediction
        model = models['autoencoder']
        model.eval()
        with torch.no_grad():
            reconstructed, encoded = model(input_tensor)
        
        # Calculate reconstruction error
        mse = F.mse_loss(reconstructed, input_tensor.view(1, -1)).item()
        
        # Return results
        return jsonify({
            'reconstruction_error': float(mse),
            'encoded_vector': encoded[0].cpu().numpy().tolist(),
            'reconstructed_image': reconstructed[0].cpu().numpy().tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/timeseries', methods=['POST'])
def predict_timeseries():
    """Time series forecasting endpoint"""
    try:
        # Get time series data
        data = request.get_json()
        
        if 'sequence' not in data:
            return jsonify({'error': 'No sequence data provided'}), 400
        
        sequence = data['sequence']
        forecast_steps = data.get('forecast_steps', 10)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1).to(device)
        
        # Make prediction
        model = models['timeseries']
        model.eval()
        
        predictions = []
        current_seq = input_tensor.clone()
        
        with torch.no_grad():
            for _ in range(forecast_steps):
                pred = model(current_seq)
                predictions.append(pred.cpu().item())
                
                # Update sequence: remove first element, add prediction
                current_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(-1)], dim=1)
        
        # Return results
        return jsonify({
            'forecast': predictions,
            'forecast_steps': forecast_steps,
            'input_length': len(sequence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/cifar10', methods=['POST'])
def predict_cifar10():
    """CIFAR-10 classification endpoint"""
    try:
        # Get image data
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((32, 32))
        image_array = np.array(image) / 255.0
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Make prediction (using classifier as proxy)
        model = models['classifier']
        model.eval()
        with torch.no_grad():
            # Flatten for classifier
            flattened = input_tensor.view(1, -1)
            output = model(flattened)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
        
        # Return results
        return jsonify({
            'predicted_class': cifar10_classes[predicted_class],
            'confidence': float(probabilities[0, predicted_class]),
            'probabilities': {
                class_name: float(prob) for class_name, prob in zip(cifar10_classes, probabilities[0].cpu().numpy())
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train/classifier', methods=['POST'])
def train_classifier():
    """Training endpoint (simplified)"""
    try:
        data = request.get_json()
        epochs = data.get('epochs', 5)
        lr = data.get('learning_rate', 0.001)
        
        # This is a simplified training example
        # In practice, you'd load real data and train properly
        model = models['classifier']
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Simulate training
        model.train()
        for epoch in range(epochs):
            # Generate dummy data for demonstration
            dummy_data = torch.randn(32, 1, 28, 28).to(device)
            dummy_targets = torch.randint(0, 10, (32,)).to(device)
            
            optimizer.zero_grad()
            output = model(dummy_data)
            loss = criterion(output, dummy_targets)
            loss.backward()
            optimizer.step()
        
        return jsonify({
            'message': f'Training completed for {epochs} epochs',
            'learning_rate': lr,
            'final_loss': float(loss)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/<model_name>/info')
def model_info(model_name):
    """Get detailed model information"""
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 404
    
    model = models[model_name]
    
    # Count parameters by layer
    layer_info = []
    for name, param in model.named_parameters():
        layer_info.append({
            'name': name,
            'shape': list(param.shape),
            'parameters': param.numel(),
            'requires_grad': param.requires_grad
        })
    
    return jsonify({
        'name': model_name,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'size_mb': sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024,
        'device': str(device),
        'layers': layer_info
    })

@app.route('/models/<model_name>/save', methods=['POST'])
def save_model(model_name):
    """Save model to file"""
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 404
    
    try:
        model = models[model_name]
        model_path = f'model_{model_name}.pth'
        torch.save(model.state_dict(), model_path)
        
        return jsonify({
            'message': f'Model {model_name} saved successfully',
            'path': model_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/<model_name>/load', methods=['POST'])
def load_model(model_name):
    """Load model from file"""
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 404
    
    try:
        data = request.get_json()
        model_path = data.get('path', f'model_{model_name}.pth')
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404
        
        model = models[model_name]
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        return jsonify({
            'message': f'Model {model_name} loaded successfully',
            'path': model_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Mini-ANNs API server...")
    print(f"Device: {device}")
    print("Available models:", list(models.keys()))
    print("API endpoints:")
    print("  GET  / - API information")
    print("  GET  /health - Health check")
    print("  GET  /models - List models")
    print("  POST /predict/classifier - Image classification")
    print("  POST /predict/autoencoder - Image reconstruction")
    print("  POST /predict/timeseries - Time series forecasting")
    print("  POST /predict/cifar10 - CIFAR-10 classification")
    print("  POST /train/classifier - Train classifier")
    print("  GET  /models/<name>/info - Model details")
    print("  POST /models/<name>/save - Save model")
    print("  POST /models/<name>/load - Load model")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
