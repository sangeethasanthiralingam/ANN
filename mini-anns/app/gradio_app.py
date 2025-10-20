"""
Gradio App for Mini-ANNs Project
Interactive dashboard for testing trained models
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import sys
import os

# Add scripts directory to path
sys.path.append('../scripts')
from utils import get_device, set_seed

# Set random seed
set_seed(42)
device = get_device()

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

# Load models (dummy models for demo)
def load_models():
    """Load pre-trained models."""
    models = {}
    
    # MNIST classifier
    mnist_model = TinyImageClassifier().to(device)
    # Load dummy weights (in real app, load from saved model)
    mnist_model.load_state_dict(torch.load('../results/logs/mnist_model.pth', map_location=device) if os.path.exists('../results/logs/mnist_model.pth') else {})
    models['mnist'] = mnist_model
    
    # CIFAR-10 classifier
    cifar_model = MiniCNN().to(device)
    # Load dummy weights
    cifar_model.load_state_dict(torch.load('../results/logs/cifar_model.pth', map_location=device) if os.path.exists('../results/logs/cifar_model.pth') else {})
    models['cifar'] = cifar_model
    
    return models

# Initialize models
models = load_models()

# Class names
mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict_mnist(image):
    """Predict MNIST digit."""
    if image is None:
        return "Please upload an image"
    
    # Preprocess image
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize
    
    # Convert to tensor
    image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    model = models['mnist']
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return f"Predicted: {mnist_classes[predicted_class]} (Confidence: {confidence:.3f})"

def predict_cifar(image):
    """Predict CIFAR-10 class."""
    if image is None:
        return "Please upload an image"
    
    # Preprocess image
    image = image.convert('RGB')  # Convert to RGB
    image = image.resize((32, 32))  # Resize to 32x32
    image = np.array(image) / 255.0  # Normalize
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Convert to tensor
    image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predict
    model = models['cifar']
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return f"Predicted: {cifar_classes[predicted_class]} (Confidence: {confidence:.3f})"

def generate_synthetic_data(model_type, num_samples):
    """Generate synthetic data for visualization."""
    if model_type == "MNIST":
        # Generate random MNIST-like data
        data = np.random.rand(num_samples, 28, 28)
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(10):
            row, col = i // 5, i % 5
            axes[row, col].imshow(data[i], cmap='gray')
            axes[row, col].set_title(f'Sample {i+1}')
            axes[row, col].axis('off')
    else:
        # Generate random CIFAR-like data
        data = np.random.rand(num_samples, 32, 32, 3)
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(10):
            row, col = i // 5, i % 5
            axes[row, col].imshow(data[i])
            axes[row, col].set_title(f'Sample {i+1}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig

def create_model_comparison():
    """Create model comparison visualization."""
    # Dummy data for demonstration
    models = ['Tiny MLP', 'Mini CNN', 'LSTM', 'Autoencoder', 'GAN']
    accuracies = [85.2, 92.1, 78.5, 88.3, 82.7]
    parameters = [50, 200, 150, 100, 300]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    ax1.bar(models, accuracies, color='skyblue')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # Parameter count comparison
    ax2.bar(models, parameters, color='lightcoral')
    ax2.set_title('Model Parameter Count')
    ax2.set_ylabel('Parameters (K)')
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(title="Mini-ANNs Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Mini-ANNs Interactive Dashboard")
        gr.Markdown("Explore neural network models and experiments from the Mini-ANNs project.")
        
        with gr.Tabs():
            # Tab 1: Image Classification
            with gr.Tab("🖼️ Image Classification"):
                gr.Markdown("## MNIST Digit Classification")
                with gr.Row():
                    with gr.Column():
                        mnist_input = gr.Image(type="pil", label="Upload MNIST Image (28x28)")
                        mnist_btn = gr.Button("Classify Digit", variant="primary")
                        mnist_output = gr.Textbox(label="Prediction Result")
                    with gr.Column():
                        gr.Markdown("### Sample MNIST Images")
                        mnist_samples = gr.Plot()
                        mnist_generate_btn = gr.Button("Generate Samples")
                
                gr.Markdown("## CIFAR-10 Object Classification")
                with gr.Row():
                    with gr.Column():
                        cifar_input = gr.Image(type="pil", label="Upload CIFAR-10 Image (32x32)")
                        cifar_btn = gr.Button("Classify Object", variant="primary")
                        cifar_output = gr.Textbox(label="Prediction Result")
                    with gr.Column():
                        gr.Markdown("### Sample CIFAR-10 Images")
                        cifar_samples = gr.Plot()
                        cifar_generate_btn = gr.Button("Generate Samples")
            
            # Tab 2: Model Comparison
            with gr.Tab("📊 Model Comparison"):
                gr.Markdown("## Model Performance Analysis")
                comparison_plot = gr.Plot()
                comparison_btn = gr.Button("Generate Comparison", variant="primary")
            
            # Tab 3: Experiment Results
            with gr.Tab("🔬 Experiment Results"):
                gr.Markdown("## Experiment Overview")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ### Available Experiments:
                        - **01**: Tiny Image Classifier
                        - **02**: Mini Autoencoder
                        - **03**: Micro LSTM
                        - **04**: Mini Time Series
                        - **05**: Anomaly Detection
                        - **06**: Mini CNN
                        - **07**: Pruning Study
                        - **08**: Toy Problems
                        - **09**: Mini GAN
                        - **10**: Energy Efficient ANN
                        - **11**: Activation Comparison
                        - **12**: Regularization Practice
                        - **13**: Learning Rate Experiments
                        - **14**: Data Size vs Accuracy
                        - **15**: Transfer Learning Mini
                        """)
                    with gr.Column():
                        gr.Markdown("""
                        ### Key Features:
                        - Interactive model testing
                        - Real-time predictions
                        - Visualization tools
                        - Performance metrics
                        - Educational content
                        """)
            
            # Tab 4: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## Mini-ANNs Project
                
                This project demonstrates various neural network architectures and techniques using PyTorch.
                
                ### Features:
                - 15 different experiments
                - Interactive web interface
                - Model comparison tools
                - Educational content
                - Real-time predictions
                
                ### Technologies:
                - PyTorch for deep learning
                - Gradio for web interface
                - Matplotlib for visualization
                - NumPy for numerical computing
                
                ### Getting Started:
                1. Upload an image to test classification
                2. Explore different model comparisons
                3. Run experiments from the notebooks
                4. Learn about neural network concepts
                """)
        
        # Event handlers
        mnist_btn.click(predict_mnist, inputs=mnist_input, outputs=mnist_output)
        cifar_btn.click(predict_cifar, inputs=cifar_input, outputs=cifar_output)
        
        mnist_generate_btn.click(lambda: generate_synthetic_data("MNIST", 10), outputs=mnist_samples)
        cifar_generate_btn.click(lambda: generate_synthetic_data("CIFAR", 10), outputs=cifar_samples)
        
        comparison_btn.click(create_model_comparison, outputs=comparison_plot)
    
    return demo

# Main function
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
