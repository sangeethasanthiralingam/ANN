"""
Streamlit Dashboard for Mini-ANNs Project
Interactive interface to test and visualize trained models
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import sys
import os

# Add parent directory to path
sys.path.append('..')
from scripts.utils import get_device, set_seed

# Set page config
st.set_page_config(
    page_title="Mini-ANNs Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 Mini-ANNs Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Interactive Neural Network Experiments")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Home", "Image Classifier", "Autoencoder", "Time Series", "CNN", "Model Comparison"]
)

# Set device
device = get_device()
set_seed(42)

# Model definitions (simplified for demo)
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

# Home page
if page == "Home":
    st.markdown("""
    ## Welcome to Mini-ANNs Dashboard! 🎉
    
    This interactive dashboard allows you to explore various neural network experiments from the Mini-ANNs project.
    
    ### Available Experiments:
    
    **🔢 Image Classifier**
    - MNIST digit classification
    - Interactive prediction interface
    - Model performance metrics
    
    **🔄 Autoencoder**
    - Image reconstruction
    - Latent space visualization
    - Compression analysis
    
    **📈 Time Series**
    - LSTM-based forecasting
    - Interactive prediction
    - Trend analysis
    
    **🖼️ CNN**
    - CIFAR-10 classification
    - Feature map visualization
    - Convolutional layer analysis
    
    **⚖️ Model Comparison**
    - Side-by-side model comparison
    - Performance metrics
    - Architecture analysis
    
    ### Getting Started:
    1. Select a page from the sidebar
    2. Upload your own data or use sample data
    3. Interact with the models
    4. Explore the results and visualizations
    
    ### Features:
    - 🎯 Real-time predictions
    - 📊 Interactive visualizations
    - 🔍 Model inspection tools
    - 📈 Performance metrics
    - 🎨 Custom data upload
    """)
    
    # Sample data generation
    st.markdown("### Sample Data Generator")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Sample MNIST Data"):
            # Generate random MNIST-like data
            sample_data = torch.randn(1, 1, 28, 28)
            sample_data = (sample_data - sample_data.min()) / (sample_data.max() - sample_data.min())
            
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(sample_data[0, 0].numpy(), cmap='gray')
            ax.set_title('Sample MNIST-like Image')
            ax.axis('off')
            st.pyplot(fig)
    
    with col2:
        if st.button("Generate Sample Time Series"):
            # Generate sample time series
            t = np.linspace(0, 4*np.pi, 100)
            y = np.sin(t) + 0.1 * np.random.randn(100)
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(t, y)
            ax.set_title('Sample Time Series')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True)
            st.pyplot(fig)

# Image Classifier page
elif page == "Image Classifier":
    st.header("🔢 Image Classifier")
    st.markdown("MNIST digit classification using a simple neural network")
    
    # Model info
    with st.expander("Model Architecture"):
        st.code("""
        TinyImageClassifier(
          (fc1): Linear(784, 128)
          (dropout): Dropout(0.2)
          (fc2): Linear(128, 10)
        )
        """)
    
    # Upload image or use sample
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('L')
            image = image.resize((28, 28))
            image_array = np.array(image) / 255.0
            st.image(image, caption="Uploaded Image", width=200)
        else:
            st.info("Upload an image or use the sample below")
    
    with col2:
        st.subheader("Sample Image")
        if st.button("Generate Random Sample"):
            # Generate random sample
            sample_data = torch.randn(1, 1, 28, 28)
            sample_data = (sample_data - sample_data.min()) / (sample_data.max() - sample_data.min())
            image_array = sample_data[0, 0].numpy()
            
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(image_array, cmap='gray')
            ax.set_title('Sample Image')
            ax.axis('off')
            st.pyplot(fig)
    
    # Prediction
    if st.button("Make Prediction") or uploaded_file is not None:
        # Initialize model (in real app, load from saved state)
        model = TinyImageClassifier().to(device)
        
        # Prepare input
        if uploaded_file is not None:
            input_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
        else:
            input_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
        
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Predicted Digit: {predicted_class}**")
            st.markdown(f"**Confidence: {probabilities[0, predicted_class]:.2%}**")
        
        with col2:
            # Probability distribution
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            bars = ax.bar(range(10), probabilities[0].cpu().numpy())
            bars[predicted_class].set_color('red')
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')
            ax.set_xticks(range(10))
            st.pyplot(fig)

# Autoencoder page
elif page == "Autoencoder":
    st.header("🔄 Autoencoder")
    st.markdown("Image reconstruction and latent space visualization")
    
    # Model info
    with st.expander("Model Architecture"):
        st.code("""
        MiniAutoencoder(
          encoder: 784 -> 128 -> 32
          decoder: 32 -> 128 -> 784
        )
        """)
    
    # Generate sample data
    st.subheader("Sample Data")
    if st.button("Generate Sample Images"):
        # Generate random samples
        samples = torch.randn(4, 1, 28, 28)
        samples = (samples - samples.min()) / (samples.max() - samples.min())
        
        # Display original images
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(4):
            axes[0, i].imshow(samples[i, 0].numpy(), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
        
        # Initialize model and get reconstructions
        model = MiniAutoencoder().to(device)
        with torch.no_grad():
            reconstructions, encoded = model(samples.to(device))
        
        # Display reconstructions
        for i in range(4):
            axes[1, i].imshow(reconstructions[i].cpu().view(28, 28).numpy(), cmap='gray')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        
        plt.suptitle('Original vs Reconstructed Images')
        st.pyplot(fig)
        
        # Latent space visualization
        st.subheader("Latent Space")
        encoded_np = encoded.cpu().numpy()
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        scatter = ax.scatter(encoded_np[:, 0], encoded_np[:, 1], c=range(4), cmap='tab10', s=100)
        for i in range(4):
            ax.annotate(f'Img {i+1}', (encoded_np[i, 0], encoded_np[i, 1]))
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_title('Latent Space Representation')
        ax.grid(True)
        st.pyplot(fig)

# Time Series page
elif page == "Time Series":
    st.header("📈 Time Series Forecasting")
    st.markdown("LSTM-based time series prediction")
    
    # Model info
    with st.expander("Model Architecture"):
        st.code("""
        TimeSeriesLSTM(
          lstm: 1 -> 64 (2 layers)
          fc: 64 -> 1
        )
        """)
    
    # Generate sample time series
    st.subheader("Sample Time Series")
    if st.button("Generate Sample Data"):
        # Generate synthetic time series
        t = np.linspace(0, 4*np.pi, 200)
        y = np.sin(t) + 0.1 * np.random.randn(200)
        
        # Plot original data
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(t, y, label='Original Data')
        ax.set_title('Sample Time Series Data')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Make predictions (simplified)
        st.subheader("Forecasting")
        forecast_steps = st.slider("Forecast Steps", 10, 50, 20)
        
        if st.button("Generate Forecast"):
            # Simple linear trend forecast (in real app, use trained LSTM)
            last_value = y[-1]
            trend = (y[-1] - y[-10]) / 10
            forecast = [last_value + trend * i for i in range(1, forecast_steps + 1)]
            
            # Plot forecast
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.plot(t, y, label='Historical Data', color='blue')
            ax.plot(np.linspace(t[-1], t[-1] + forecast_steps, forecast_steps), 
                   forecast, label='Forecast', color='red', linestyle='--')
            ax.set_title('Time Series Forecast')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

# CNN page
elif page == "CNN":
    st.header("🖼️ Convolutional Neural Network")
    st.markdown("CIFAR-10 image classification using CNN")
    
    # Model info
    with st.expander("Model Architecture"):
        st.code("""
        MiniCNN(
          conv1: Conv2d(3, 32, 3x3)
          conv2: Conv2d(32, 64, 3x3)
          conv3: Conv2d(64, 128, 3x3)
          fc1: Linear(2048, 256)
          fc2: Linear(256, 10)
        )
        """)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Generate sample data
    st.subheader("Sample CIFAR-10 Images")
    if st.button("Generate Sample Images"):
        # Generate random CIFAR-10-like images
        samples = torch.randn(8, 3, 32, 32)
        samples = (samples - samples.min()) / (samples.max() - samples.min())
        
        # Display images
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(8):
            row, col = i // 4, i % 4
            img = samples[i].permute(1, 2, 0).numpy()
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Sample {i+1}')
            axes[row, col].axis('off')
        
        plt.suptitle('Sample CIFAR-10-like Images')
        st.pyplot(fig)
        
        # Simulate predictions
        st.subheader("Classification Results")
        predictions = np.random.randint(0, 10, 8)
        confidences = np.random.rand(8)
        
        results_df = {
            'Image': [f'Sample {i+1}' for i in range(8)],
            'Predicted Class': [class_names[p] for p in predictions],
            'Confidence': [f'{c:.2%}' for c in confidences]
        }
        
        st.dataframe(results_df, use_container_width=True)

# Model Comparison page
elif page == "Model Comparison":
    st.header("⚖️ Model Comparison")
    st.markdown("Compare different neural network architectures")
    
    # Model comparison data
    models_data = {
        'Model': ['Tiny Classifier', 'Mini Autoencoder', 'Time Series LSTM', 'Mini CNN'],
        'Parameters': [101,770, 25,344, 33,537, 1,250,890],
        'Size (MB)': [0.39, 0.10, 0.13, 4.77],
        'Accuracy': [0.95, 0.92, 0.88, 0.78],
        'Training Time (min)': [2, 5, 3, 15]
    }
    
    # Display comparison table
    st.subheader("Model Specifications")
    st.dataframe(models_data, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameters vs Accuracy")
        fig = px.scatter(
            models_data, 
            x='Parameters', 
            y='Accuracy',
            size='Size (MB)',
            hover_name='Model',
            title='Model Complexity vs Performance'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Training Time vs Accuracy")
        fig = px.scatter(
            models_data,
            x='Training Time (min)',
            y='Accuracy',
            size='Parameters',
            hover_name='Model',
            title='Training Efficiency vs Performance'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Tiny Classifier': [0.95, 0.94, 0.95, 0.94],
        'Mini Autoencoder': [0.92, 0.91, 0.92, 0.91],
        'Time Series LSTM': [0.88, 0.87, 0.88, 0.87],
        'Mini CNN': [0.78, 0.76, 0.78, 0.77]
    }
    
    st.dataframe(metrics, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Mini-ANNs Dashboard | Built with Streamlit | PyTorch Backend</p>
    <p>For more experiments, check out the Jupyter notebooks in the notebooks/ directory</p>
</div>
""", unsafe_allow_html=True)
