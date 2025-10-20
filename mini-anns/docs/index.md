# Mini-ANNs Documentation

Welcome to the Mini-ANNs project documentation! This comprehensive guide will help you understand, set up, and use the various neural network experiments in this project.

## 📚 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Experiments Guide](#experiments-guide)
- [API Documentation](#api-documentation)
- [Dashboard Usage](#dashboard-usage)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

Mini-ANNs is a comprehensive collection of minimal artificial neural network experiments designed for educational purposes. Each experiment demonstrates specific concepts in deep learning using PyTorch, with working code that runs without modification.

### Key Features

- **15 Complete Experiments**: From basic classification to advanced GANs
- **Interactive Dashboard**: Streamlit-based web interface
- **RESTful API**: Flask-based API for model inference
- **Comprehensive Documentation**: Detailed explanations and tutorials
- **Ready-to-Run Code**: All experiments work out of the box
- **Educational Focus**: Clear explanations and learning objectives

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Jupyter Notebook
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd mini-anns
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv mini-anns-env
   source mini-anns-env/bin/activate  # On Windows: mini-anns-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run your first experiment:**
   ```bash
   jupyter notebook notebooks/01_tiny_image_classifier.ipynb
   ```

### Quick Test

```python
# Test installation
import torch
import torchvision
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 📁 Project Structure

```
mini-anns/
├── README.md                 # Main project documentation
├── requirements.txt          # Python dependencies
├── data/                    # Dataset storage
│   ├── mnist/              # MNIST data
│   ├── fashion-mnist/      # Fashion-MNIST data
│   └── cifar10/            # CIFAR-10 data
├── notebooks/              # Jupyter notebooks
│   ├── 01_tiny_image_classifier.ipynb
│   ├── 02_mini_autoencoder.ipynb
│   ├── 03_micro_lstm.ipynb
│   ├── 04_mini_time_series.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_mini_cnn.ipynb
│   ├── 07_pruning_study.ipynb
│   ├── 08_toy_problems.ipynb
│   ├── 09_mini_gan.ipynb
│   ├── 10_energy_efficient_ann.ipynb
│   ├── 11_activation_comparison.ipynb
│   ├── 12_regularization_practice.ipynb
│   ├── 13_learning_rate_experiments.ipynb
│   ├── 14_data_size_vs_accuracy.ipynb
│   └── 15_transfer_learning_mini.ipynb
├── scripts/                # Utility scripts
│   ├── utils.py           # Helper functions
│   ├── train.py           # Training utilities
│   └── evaluate.py        # Evaluation utilities
├── app/                   # Web applications
│   ├── streamlit_app.py   # Streamlit dashboard
│   └── gradio_app.py      # Gradio interface
├── api/                   # API server
│   ├── app.py            # Flask API
│   └── model.pth         # Example saved model
├── docs/                  # Documentation
│   ├── index.md          # This file
│   └── experiments.md    # Detailed experiment guide
└── results/              # Output directory
    ├── plots/            # Generated plots
    └── logs/             # Training logs
```

## 🧪 Experiments Guide

### Core Experiments (1-10)

1. **Tiny Image Classifier** - Basic MNIST classification
2. **Mini Autoencoder** - Image compression and reconstruction
3. **Micro LSTM** - Character-level text generation
4. **Mini Time Series** - LSTM-based forecasting
5. **Anomaly Detection** - VAE for outlier detection
6. **Mini CNN** - Convolutional neural networks
7. **Pruning Study** - Network compression techniques
8. **Toy Problems** - Classic ML problems (XOR, spirals)
9. **Mini GAN** - Generative adversarial networks
10. **Energy Efficient ANN** - Model optimization

### Enhancement Experiments (11-15)

11. **Activation Comparison** - ReLU vs Sigmoid vs Tanh
12. **Regularization Practice** - Dropout, L2, early stopping
13. **Learning Rate Experiments** - LR scheduling and optimization
14. **Data Size vs Accuracy** - Dataset size impact analysis
15. **Transfer Learning Mini** - Fine-tuning pre-trained models

### Running Experiments

Each notebook is self-contained and can be run independently:

```bash
# Run specific notebook
jupyter notebook notebooks/01_tiny_image_classifier.ipynb

# Run all notebooks (in sequence)
for notebook in notebooks/*.ipynb; do
    jupyter nbconvert --execute --to notebook --inplace "$notebook"
done
```

## 🌐 API Documentation

### Starting the API Server

```bash
cd api
python app.py
```

The API will be available at `http://localhost:5000`

### Available Endpoints

#### General Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /models` - List available models

#### Prediction Endpoints
- `POST /predict/classifier` - Image classification
- `POST /predict/autoencoder` - Image reconstruction
- `POST /predict/timeseries` - Time series forecasting
- `POST /predict/cifar10` - CIFAR-10 classification

#### Model Management
- `GET /models/<name>/info` - Model details
- `POST /models/<name>/save` - Save model
- `POST /models/<name>/load` - Load model

### Example API Usage

```python
import requests
import base64
from PIL import Image
import io

# Prepare image
image = Image.open('sample.png').convert('L').resize((28, 28))
buffer = io.BytesIO()
image.save(buffer, format='PNG')
image_data = base64.b64encode(buffer.getvalue()).decode()

# Make prediction
response = requests.post('http://localhost:5000/predict/classifier', 
                        json={'image': image_data})
result = response.json()
print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🎨 Dashboard Usage

### Streamlit Dashboard

```bash
cd app
streamlit run streamlit_app.py
```

The dashboard will be available at `http://localhost:8501`

### Features

- **Interactive Model Testing** - Upload images and get predictions
- **Real-time Visualizations** - Dynamic plots and charts
- **Model Comparison** - Side-by-side performance analysis
- **Data Exploration** - Interactive data visualization
- **Custom Data Upload** - Test with your own datasets

### Gradio Interface

```bash
cd app
python gradio_app.py
```

Alternative interface with different UI components.

## 🔧 Advanced Usage

### Custom Model Training

```python
from scripts.train import train_model
from scripts.utils import load_mnist_data

# Load data
train_loader, val_loader, test_loader = load_mnist_data()

# Define custom model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train model
model = MyModel()
trainer = train_model(model, train_loader, val_loader, 
                     task_type='classification', epochs=10)
```

### Custom Evaluation

```python
from scripts.evaluate import evaluate_model

# Evaluate model
results = evaluate_model(model, test_loader, 
                        task_type='classification',
                        save_dir='results/plots/my_model')

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")
```

### Model Saving and Loading

```python
import torch

# Save model
torch.save(model.state_dict(), 'my_model.pth')

# Load model
model = MyModel()
model.load_state_dict(torch.load('my_model.pth'))
model.eval()
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Adding New Experiments

1. Create a new notebook in `notebooks/`
2. Follow the naming convention: `XX_experiment_name.ipynb`
3. Include:
   - Clear problem statement
   - Dataset loading code
   - Model implementation
   - Training loop
   - Evaluation and visualization
4. Update documentation

### Reporting Issues

- Use GitHub Issues
- Include error messages and system information
- Provide minimal reproducible examples

### Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to functions
- Include comments for complex logic

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Use CPU instead
device = torch.device('cpu')
```

**Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

**Dataset Download Issues**
```python
# Set download=True and check internet connection
dataset = torchvision.datasets.MNIST('data', download=True)
```

**Jupyter Kernel Issues**
```bash
# Restart kernel and clear output
# Or restart Jupyter server
```

### Performance Tips

- Use GPU when available
- Adjust batch sizes based on memory
- Use mixed precision training for large models
- Enable cuDNN optimizations

### Getting Help

- Check the [experiments guide](experiments.md)
- Review error messages carefully
- Search existing issues
- Ask questions in discussions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- The machine learning community for educational resources
- Contributors who help improve these experiments

---

**Happy Learning! 🎓**

Start with the first notebook and work your way through. Each experiment builds upon previous concepts while introducing new techniques and applications.
