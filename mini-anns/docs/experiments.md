# Experiments Guide

This document provides detailed information about each experiment in the Mini-ANNs project.

## 📋 Table of Contents

- [Core Experiments (1-10)](#core-experiments-1-10)
- [Enhancement Experiments (11-15)](#enhancement-experiments-11-15)
- [Experiment Templates](#experiment-templates)
- [Best Practices](#best-practices)

## 🧪 Core Experiments (1-10)

### 01. Tiny Image Classifier

**File:** `01_tiny_image_classifier.ipynb`

**Objective:** Classify MNIST digits using a minimal fully connected network

**Key Concepts:**
- Basic neural network architecture
- Image preprocessing
- Classification training
- Model evaluation

**Model Architecture:**
```
Input (784) → FC (128) → ReLU → Dropout → FC (10) → Output
```

**Expected Results:**
- Training accuracy: ~95%
- Test accuracy: ~94%
- Training time: ~2 minutes

**Learning Outcomes:**
- Understanding feedforward networks
- Image data preprocessing
- Training loop implementation
- Evaluation metrics

---

### 02. Mini Autoencoder

**File:** `02_mini_autoencoder.ipynb`

**Objective:** Learn compressed representations of MNIST images

**Key Concepts:**
- Encoder-decoder architecture
- Unsupervised learning
- Dimensionality reduction
- Reconstruction quality

**Model Architecture:**
```
Encoder: 784 → 128 → 32
Decoder: 32 → 128 → 784
```

**Expected Results:**
- Compression ratio: 24.5x (784 → 32)
- Reconstruction error: <0.05
- Latent space clustering by digit

**Learning Outcomes:**
- Autoencoder principles
- Latent space visualization
- Reconstruction metrics
- Unsupervised learning

---

### 03. Micro LSTM

**File:** `03_micro_lstm.ipynb`

**Objective:** Generate text using a minimal LSTM

**Key Concepts:**
- Recurrent neural networks
- Character-level processing
- Sequence generation
- Temperature sampling

**Model Architecture:**
```
Embedding → LSTM (64) → FC (vocab_size)
```

**Expected Results:**
- Text generation capability
- Learning character patterns
- Temperature-controlled sampling

**Learning Outcomes:**
- LSTM architecture
- Sequence modeling
- Text generation
- Sampling strategies

---

### 04. Mini Time Series

**File:** `04_mini_time_series.ipynb`

**Objective:** Predict future values in synthetic time series data

**Key Concepts:**
- Time series forecasting
- LSTM for sequences
- Multi-step prediction
- Trend analysis

**Model Architecture:**
```
LSTM (64, 2 layers) → FC (1)
```

**Expected Results:**
- RMSE: <0.2
- Captures seasonal patterns
- Multi-step forecasting

**Learning Outcomes:**
- Time series preprocessing
- LSTM for forecasting
- Evaluation metrics
- Future prediction

---

### 05. Anomaly Detection

**File:** `05_anomaly_detection.ipynb`

**Objective:** Detect anomalies using a Variational Autoencoder

**Key Concepts:**
- Variational Autoencoders
- Anomaly detection
- Reconstruction error
- ROC analysis

**Model Architecture:**
```
Encoder: 2 → 32 → μ, σ
Decoder: z → 32 → 2
```

**Expected Results:**
- AUC: >0.9
- Clear separation of normal/anomalous
- Threshold optimization

**Learning Outcomes:**
- VAE principles
- Anomaly detection techniques
- ROC curve analysis
- Threshold selection

---

### 06. Mini CNN

**File:** `06_mini_cnn.ipynb`

**Objective:** Image classification using convolutional layers

**Key Concepts:**
- Convolutional operations
- Feature extraction
- Pooling layers
- Feature map visualization

**Model Architecture:**
```
Conv2d(3,32) → MaxPool → Conv2d(32,64) → MaxPool → Conv2d(64,128) → MaxPool → FC(256) → FC(10)
```

**Expected Results:**
- CIFAR-10 accuracy: ~75%
- Feature map visualization
- Convolutional layer analysis

**Learning Outcomes:**
- CNN architecture
- Convolutional operations
- Feature visualization
- Image classification

---

### 07. Pruning Study

**File:** `07_pruning_study.ipynb`

**Objective:** Explore network pruning techniques

**Key Concepts:**
- Network pruning
- Magnitude-based pruning
- Sparsity analysis
- Accuracy vs. compression

**Model Architecture:**
```
Prunable MLP with magnitude-based pruning
```

**Expected Results:**
- 50% sparsity with <5% accuracy loss
- Compression vs. accuracy trade-offs
- Pruning schedule analysis

**Learning Outcomes:**
- Model compression
- Pruning techniques
- Sparsity analysis
- Efficiency trade-offs

---

### 08. Toy Problems

**File:** `08_toy_problems.ipynb`

**Objective:** Solve classic ML problems with minimal networks

**Key Concepts:**
- XOR problem
- Non-linear decision boundaries
- Spiral and moons datasets
- Network capacity

**Model Architecture:**
```
Various architectures for different problems
```

**Expected Results:**
- XOR: 100% accuracy
- Spirals: >95% accuracy
- Decision boundary visualization

**Learning Outcomes:**
- Classic ML problems
- Non-linear separability
- Decision boundaries
- Network capacity

---

### 09. Mini GAN

**File:** `09_mini_gan.ipynb`

**Objective:** Generate synthetic data using GANs

**Key Concepts:**
- Generative Adversarial Networks
- Generator and discriminator
- Adversarial training
- Generated sample quality

**Model Architecture:**
```
Generator: Noise → FC → Generated Data
Discriminator: Data → FC → Real/Fake
```

**Expected Results:**
- Generated sample quality
- Training dynamics
- Loss curve analysis

**Learning Outcomes:**
- GAN principles
- Adversarial training
- Generator design
- Sample quality assessment

---

### 10. Energy Efficient ANN

**File:** `10_energy_efficient_ann.ipynb`

**Objective:** Explore energy-efficient neural network designs

**Key Concepts:**
- Model quantization
- Energy consumption estimation
- Accuracy vs. efficiency
- Optimization techniques

**Model Architecture:**
```
Quantized and pruned networks
```

**Expected Results:**
- Energy consumption reduction
- Accuracy vs. efficiency trade-offs
- Quantization analysis

**Learning Outcomes:**
- Model efficiency
- Quantization techniques
- Energy optimization
- Performance trade-offs

---

## 🔬 Enhancement Experiments (11-15)

### 11. Activation Comparison

**File:** `11_activation_comparison.ipynb`

**Objective:** Compare ReLU vs Sigmoid vs Tanh activations

**Key Concepts:**
- Activation function properties
- Training dynamics
- Gradient flow
- Convergence behavior

**Expected Results:**
- ReLU: Best convergence
- Sigmoid: Gradient vanishing
- Tanh: Moderate performance

**Learning Outcomes:**
- Activation function effects
- Training dynamics
- Gradient analysis
- Convergence comparison

---

### 12. Regularization Practice

**File:** `12_regularization_practice.ipynb`

**Objective:** Explore regularization techniques

**Key Concepts:**
- Dropout
- L2 weight decay
- Early stopping
- Batch normalization

**Expected Results:**
- Improved generalization
- Reduced overfitting
- Regularization comparison

**Learning Outcomes:**
- Regularization techniques
- Overfitting prevention
- Generalization improvement
- Technique comparison

---

### 13. Learning Rate Experiments

**File:** `13_learning_rate_experiments.ipynb`

**Objective:** Compare different learning rates and schedules

**Key Concepts:**
- Learning rate schedules
- Optimization algorithms
- Convergence analysis
- Hyperparameter tuning

**Expected Results:**
- Optimal learning rate identification
- Schedule comparison
- Convergence analysis

**Learning Outcomes:**
- Learning rate effects
- Schedule strategies
- Optimization algorithms
- Hyperparameter tuning

---

### 14. Data Size vs Accuracy

**File:** `14_data_size_vs_accuracy.ipynb`

**Objective:** Analyze dataset size impact on accuracy

**Key Concepts:**
- Dataset size effects
- Learning curves
- Data efficiency
- Sample complexity

**Expected Results:**
- Learning curve analysis
- Data efficiency metrics
- Sample complexity insights

**Learning Outcomes:**
- Data size effects
- Learning curves
- Data efficiency
- Sample complexity

---

### 15. Transfer Learning Mini

**File:** `15_transfer_learning_mini.ipynb`

**Objective:** Fine-tune pre-trained models on small datasets

**Key Concepts:**
- Transfer learning
- Fine-tuning
- Pre-trained models
- Domain adaptation

**Expected Results:**
- Improved performance on small datasets
- Faster convergence
- Transfer learning benefits

**Learning Outcomes:**
- Transfer learning principles
- Fine-tuning techniques
- Pre-trained model usage
- Domain adaptation

---

## 📝 Experiment Templates

### Basic Template Structure

```python
# 1. Imports and Setup
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../scripts')
from utils import get_device, set_seed

# 2. Set random seed and device
set_seed(42)
device = get_device()

# 3. Load and visualize data
# Data loading code here

# 4. Define model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Model layers
    
    def forward(self, x):
        # Forward pass
        return x

# 5. Train model
# Training loop here

# 6. Evaluate model
# Evaluation code here

# 7. Visualize results
# Visualization code here
```

### Evaluation Template

```python
# Model evaluation
def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy
```

### Visualization Template

```python
# Training history visualization
def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## 🎯 Best Practices

### Code Organization

1. **Clear Structure**: Follow the template structure
2. **Comments**: Explain complex logic
3. **Docstrings**: Document functions and classes
4. **Naming**: Use descriptive variable names

### Experiment Design

1. **Single Purpose**: Each experiment should focus on one concept
2. **Reproducible**: Use random seeds and fixed parameters
3. **Educational**: Include clear explanations
4. **Complete**: All code should run without modification

### Visualization

1. **Clear Labels**: Always label axes and add titles
2. **Consistent Style**: Use consistent colors and styles
3. **Save Plots**: Save important visualizations
4. **Interactive**: Use interactive plots when possible

### Documentation

1. **Markdown Cells**: Use markdown for explanations
2. **Code Comments**: Comment complex code sections
3. **Results**: Document expected and actual results
4. **Learning Outcomes**: List what students will learn

### Performance

1. **Efficient Code**: Use vectorized operations
2. **Memory Management**: Clear unnecessary variables
3. **GPU Usage**: Use GPU when available
4. **Batch Processing**: Process data in batches

### Error Handling

1. **Try-Catch**: Handle potential errors gracefully
2. **Validation**: Validate inputs and outputs
3. **Logging**: Log important information
4. **Debugging**: Include debugging information

---

This guide should help you understand and work with each experiment in the Mini-ANNs project. Each experiment is designed to be educational, reproducible, and immediately runnable.
