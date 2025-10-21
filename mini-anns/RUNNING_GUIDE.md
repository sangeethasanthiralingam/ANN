# 🚀 Mini-ANNs Running Guide

Complete guide on how to run and use the Mini-ANNs project.

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Running Experiments](#-running-experiments)
- [Web Applications](#-web-applications)
- [API Server](#-api-server)
- [Troubleshooting](#-troubleshooting)
- [Results & Outputs](#-results--outputs)

## ⚡ Quick Start

### 1. **Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt
```

### 2. **Run Your First Experiment**
```bash
# Navigate to notebooks directory
cd mini-anns/notebooks

# Run Experiment 01: Tiny Image Classifier
python run_experiment_01.py
```

### 3. **Start API Server**
```bash
# In a new terminal
cd mini-anns
python app.py
# Server will be available at: http://localhost:5000
```

## 🔧 Installation

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- pip package manager
- Git (for cloning)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd mini-anns
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   # Create virtual environment
   python -m venv mini-anns-env
   
   # Activate virtual environment
   # On Windows:
   mini-anns-env\Scripts\activate
   # On macOS/Linux:
   source mini-anns-env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   python -c "import torchvision; print('Torchvision version:', torchvision.__version__)"
   ```

## 🧪 Running Experiments

### Method 1: Direct Python Scripts (Recommended)

**Run Individual Experiments:**
```bash
cd mini-anns/notebooks

# Experiment 01: Tiny Image Classifier
python run_experiment_01.py

# Experiment 02: Mini Autoencoder (create script first)
python run_experiment_02.py

# Experiment 03: Micro LSTM (create script first)
python run_experiment_03.py
```

**Expected Output:**
```
🚀 Starting Mini-ANNs Experiment 01: Tiny Image Classifier
============================================================
📱 Using device: cpu
📊 Loading MNIST dataset...
✅ Training samples: 60000
✅ Test samples: 10000
🧠 Model parameters: 101,770

🎯 Starting training...
Epoch 1/3 - Train Loss: 0.4252, Train Acc: 87.28%, Test Acc: 93.10%
Epoch 2/3 - Train Loss: 0.2316, Train Acc: 93.00%, Test Acc: 95.08%
Epoch 3/3 - Train Loss: 0.1893, Train Acc: 94.34%, Test Acc: 95.54%

🎉 Experiment 01 Complete!
📊 Final Test Accuracy: 95.54%
💾 Results saved to: ../results/plots/
```

### Method 2: Jupyter Notebooks

**Start Jupyter:**
```bash
# Option 1: Using jupyter command
jupyter notebook

# Option 2: Using python module
python -m notebook

# Option 3: Using jupyterlab
jupyter lab
```

**Access Notebooks:**
- Open browser to: http://localhost:8888
- Navigate to `notebooks/` folder
- Open any experiment notebook
- Run all cells (Shift+Enter)

### Method 3: Convert Notebooks to Scripts

**Convert any notebook to Python script:**
```bash
# Convert notebook to Python
jupyter nbconvert --to python notebooks/01_tiny_image_classifier.ipynb

# Run the converted script
python notebooks/01_tiny_image_classifier.py
```

## 🌐 Web Applications

### Streamlit Dashboard

**Start Streamlit:**
```bash
cd mini-anns/app
streamlit run streamlit_app.py
```

**Access Dashboard:**
- URL: http://localhost:8501
- Features: Interactive model demos, real-time predictions
- Upload images to test models

### Gradio Interface

**Start Gradio:**
```bash
cd mini-anns/app
python gradio_app.py
```

**Access Interface:**
- URL: http://localhost:7860
- Features: Drag-and-drop interface, model comparison
- Interactive widgets for parameter tuning

## 🔌 API Server

### Start Flask API Server

**Basic Server:**
```bash
cd mini-anns
python app.py
```

**Server Output:**
```
🚀 Starting Mini-ANNs API Server...
📱 Device: cpu
🌐 Server will be available at: http://localhost:5000
📚 API Documentation: http://localhost:5000
🔍 Health Check: http://localhost:5000/health
📋 Available Models: http://localhost:5000/models
==================================================
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.8.174:5000
```

### API Endpoints

**Available Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /models` - Available models
- `POST /predict/mnist` - MNIST digit classification
- `POST /predict/cifar` - CIFAR-10 object classification
- `POST /predict/synthetic` - Generate synthetic predictions

**Test API:**
```bash
# Health check
curl http://localhost:5000/health

# Get model information
curl http://localhost:5000/models

# Test MNIST prediction (with base64 image)
curl -X POST http://localhost:5000/predict/mnist \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'
```

## 📊 Results & Outputs

### Generated Files

**Plots and Visualizations:**
```
mini-anns/results/plots/
├── experiment_01_results.png      # Training/validation curves
├── experiment_01_predictions.png   # Sample predictions
├── experiment_02_reconstruction.png # Autoencoder reconstructions
└── ...
```

**Training Logs:**
```
mini-anns/results/logs/
├── experiment_01_log.csv          # Training metrics
├── experiment_02_log.csv          # Loss and accuracy over time
└── ...
```

**Saved Models:**
```
mini-anns/results/models/
├── experiment_01_model.pth        # Trained model weights
├── experiment_02_model.pth        # Autoencoder weights
└── ...
```

### Viewing Results

**View Plots:**
```bash
# Open results directory
cd mini-anns/results/plots
# View generated PNG files
```

**Analyze Logs:**
```bash
# View training logs
cd mini-anns/results/logs
# Open CSV files in Excel or pandas
```

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Solution: Install missing packages
pip install torch torchvision matplotlib pandas scikit-learn jupyter
```

**2. CUDA/GPU Issues**
```python
# Force CPU usage in notebooks
device = torch.device('cpu')
```

**3. Memory Issues**
```python
# Reduce batch size
batch_size = 32  # Instead of 64
```

**4. Jupyter Not Starting**
```bash
# Try different methods
python -m notebook
python -m jupyter notebook
jupyter notebook
```

**5. Port Already in Use**
```bash
# Kill process using port
netstat -ano | findstr :5000
taskkill /PID <process_id> /F
```

### PowerShell Issues (Windows)

**Problem: `&&` not recognized**
```powershell
# Instead of: cd folder && command
# Use: cd folder; command
cd mini-anns; python app.py
```

**Problem: `mkdir -p` not working**
```powershell
# Instead of: mkdir -p folder
# Use: New-Item -ItemType Directory -Path folder
New-Item -ItemType Directory -Path results/plots
```

### Performance Optimization

**For Faster Training:**
```python
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Increase batch size
batch_size = 128

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
```

**For Memory Efficiency:**
```python
# Reduce model size
hidden_size = 64  # Instead of 128

# Use gradient checkpointing
torch.utils.checkpoint.checkpoint_sequential
```

## 📈 Monitoring & Logging

### TensorBoard Integration

**Start TensorBoard:**
```bash
tensorboard --logdir=mini-anns/results/logs
```

**View Metrics:**
- URL: http://localhost:6006
- Features: Loss curves, accuracy plots, model graphs

### CSV Logging

**Training Metrics:**
```python
import pandas as pd

# Log training metrics
metrics = {
    'epoch': epoch,
    'train_loss': train_loss,
    'train_acc': train_acc,
    'test_acc': test_acc
}
pd.DataFrame([metrics]).to_csv('results/logs/experiment_01.csv', mode='a', header=False)
```

## 🚀 Advanced Usage

### Running All Experiments

**Batch Execution:**
```bash
# Create a script to run all experiments
python run_all_experiments.py
```

**Parallel Execution:**
```bash
# Run multiple experiments in parallel
python run_experiment_01.py &
python run_experiment_02.py &
python run_experiment_03.py &
```

### Custom Experiments

**Create New Experiment:**
```python
# Copy template
cp run_experiment_01.py run_experiment_16.py

# Modify for your experiment
# Update model, dataset, training loop
```

### Integration with Other Tools

**Weights & Biases:**
```python
import wandb
wandb.init(project="mini-anns")
wandb.log({"accuracy": accuracy})
```

**MLflow:**
```python
import mlflow
mlflow.start_run()
mlflow.log_param("learning_rate", 0.001)
mlflow.log_metric("accuracy", accuracy)
```

## 📞 Getting Help

### Documentation
- **Main README**: `README.md`
- **Experiment Guide**: `docs/experiments.md`
- **API Documentation**: `docs/api.md`

### Support
- **Issues**: Open GitHub issue
- **Discussions**: Use GitHub discussions
- **Email**: Contact project maintainer

### Community
- **Discord**: Join our Discord server
- **Reddit**: r/MachineLearning
- **Stack Overflow**: Tag questions with `mini-anns`

---

## 🎯 Quick Reference

### Essential Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run first experiment
cd mini-anns/notebooks && python run_experiment_01.py

# Start API server
cd mini-anns && python app.py

# Start Streamlit dashboard
cd mini-anns/app && streamlit run streamlit_app.py

# Start Jupyter
jupyter notebook
```

### Key URLs
- **API Server**: http://localhost:5000
- **Streamlit**: http://localhost:8501
- **Gradio**: http://localhost:7860
- **Jupyter**: http://localhost:8888
- **TensorBoard**: http://localhost:6006

### File Locations
- **Experiments**: `mini-anns/notebooks/`
- **Results**: `mini-anns/results/`
- **Web Apps**: `mini-anns/app/`
- **API**: `mini-anns/api/`

**Happy Experimenting! 🧠✨**
