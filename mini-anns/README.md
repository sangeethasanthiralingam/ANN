# Mini-ANNs: A Collection of Minimal Artificial Neural Network Experiments

A comprehensive collection of 15 minimal artificial neural network experiments implemented in PyTorch. Each experiment is designed to be educational, self-contained, and immediately runnable.

## 🚀 Quick Start

1. **Clone and setup:**
   ```bash
   git clone <your-repo-url>
   cd mini-anns
   pip install -r requirements.txt
   ```

2. **Run any experiment:**
   ```bash
   jupyter notebook notebooks/01_tiny_image_classifier.ipynb
   ```

3. **Launch interactive dashboard:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

4. **Start API server:**
   ```bash
   python api/app.py
   ```

## 📚 Experiments Overview

### Core Experiments (1-10)

### 1. **Tiny Image Classifier** (`01_tiny_image_classifier.ipynb`)
- **Goal**: Classify MNIST digits using a minimal fully connected network
- **Model**: 2-layer MLP (784 → 128 → 10)
- **Features**: Basic training loop, accuracy tracking, confusion matrix

### 2. **Mini Autoencoder** (`02_mini_autoencoder.ipynb`)
- **Goal**: Learn compressed representations of MNIST images
- **Model**: Encoder-decoder architecture (784 → 32 → 784)
- **Features**: Reconstruction visualization, latent space exploration

### 3. **Micro LSTM** (`03_micro_lstm.ipynb`)
- **Goal**: Generate text using a minimal LSTM
- **Model**: Single-layer LSTM with embedding layer
- **Features**: Character-level text generation, training on simple sequences

### 4. **Mini Time Series** (`04_mini_time_series.ipynb`)
- **Goal**: Predict future values in synthetic time series data
- **Model**: LSTM-based sequence predictor
- **Features**: Sine wave prediction, trend analysis, multiple forecasting horizons

### 5. **Anomaly Detection** (`05_anomaly_detection.ipynb`)
- **Goal**: Detect anomalies in multivariate data using autoencoders
- **Model**: Variational Autoencoder (VAE)
- **Features**: Anomaly scoring, threshold optimization, ROC curves

### 6. **Mini CNN** (`06_mini_cnn.ipynb`)
- **Goal**: Image classification using convolutional layers
- **Model**: Simple CNN (Conv2D → MaxPool → FC)
- **Features**: CIFAR-10 classification, feature map visualization

### 7. **Pruning Study** (`07_pruning_study.ipynb`)
- **Goal**: Explore network pruning techniques
- **Model**: Prunable MLP with magnitude-based pruning
- **Features**: Pruning schedules, sparsity analysis, accuracy vs. compression

### 8. **Toy Problems** (`08_toy_problems.ipynb`)
- **Goal**: Solve classic ML problems with minimal networks
- **Model**: Various architectures (XOR, spiral, moons)
- **Features**: Decision boundary visualization, convergence analysis

### 9. **Mini GAN** (`09_mini_gan.ipynb`)
- **Goal**: Generate synthetic data using Generative Adversarial Networks
- **Model**: Generator + Discriminator with simple architectures
- **Features**: Training dynamics, generated sample quality, loss curves

### 10. **Energy Efficient ANN** (`10_energy_efficient_ann.ipynb`)
- **Goal**: Explore energy-efficient neural network designs
- **Model**: Quantized and pruned networks
- **Features**: Energy consumption estimation, accuracy-efficiency trade-offs

### Enhancement Experiments (11-15)

### 11. **Activation Comparison** (`11_activation_comparison.ipynb`)
- **Goal**: Compare ReLU vs Sigmoid vs Tanh activation functions
- **Model**: MLP with different activations
- **Features**: Training dynamics, convergence analysis, gradient flow

### 12. **Regularization Practice** (`12_regularization_practice.ipynb`)
- **Goal**: Explore regularization techniques
- **Model**: MLP with dropout, L2, early stopping
- **Features**: Overfitting prevention, generalization improvement

### 13. **Learning Rate Experiments** (`13_learning_rate_experiments.ipynb`)
- **Goal**: Compare different learning rates and schedules
- **Model**: MLP with various optimizers
- **Features**: LR scheduling, convergence analysis, hyperparameter tuning

### 14. **Data Size vs Accuracy** (`14_data_size_vs_accuracy.ipynb`)
- **Goal**: Analyze dataset size impact on accuracy
- **Model**: MLP trained on different data sizes
- **Features**: Learning curves, data efficiency, sample complexity

### 15. **Transfer Learning Mini** (`15_transfer_learning_mini.ipynb`)
- **Goal**: Fine-tune pre-trained models on small datasets
- **Model**: Pre-trained CNN with fine-tuning
- **Features**: Transfer learning, domain adaptation, small dataset training

## 🛠️ Project Structure

```
mini-anns/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                       # Dataset storage (auto-downloaded)
│   ├── mnist/                 # MNIST data
│   ├── fashion-mnist/         # Fashion-MNIST data
│   └── cifar10/               # CIFAR-10 data
├── notebooks/                  # 15 Jupyter notebooks with experiments
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
├── scripts/                    # Reusable utility scripts
│   ├── utils.py               # Helper functions
│   ├── train.py               # Generic training loop
│   └── evaluate.py            # Evaluation utilities
├── app/                       # Web applications
│   ├── streamlit_app.py       # Streamlit dashboard
│   └── gradio_app.py          # Gradio interface
├── api/                       # API server
│   ├── app.py                 # Flask API
│   └── model.pth              # Example saved model
├── docs/                      # Documentation
│   ├── index.md               # Main documentation
│   └── experiments.md         # Detailed experiment guide
└── results/                   # Output directory
    ├── plots/                 # Generated plots and visualizations
    └── logs/                  # Training logs and metrics
```

## 🌟 Enhancements

### Interactive Dashboard
- **Streamlit App**: `streamlit run app/streamlit_app.py`
- **Gradio Interface**: `python app/gradio_app.py`
- Features: Real-time predictions, interactive visualizations, model comparison

### RESTful API
- **Flask Server**: `python api/app.py`
- Endpoints: Image classification, time series forecasting, model management
- Features: Model inference, training, saving/loading

### Comprehensive Documentation
- **Main Docs**: `docs/index.md`
- **Experiment Guide**: `docs/experiments.md`
- Features: Detailed explanations, code templates, best practices

### Advanced Features
- **Model Saving/Loading**: Automatic model persistence
- **Metrics Logging**: CSV and TensorBoard integration
- **Visualization**: Interactive plots and charts
- **Custom Data**: Support for user-uploaded datasets

## 🔧 Dependencies

- **PyTorch**: Deep learning framework
- **Torchvision**: Computer vision datasets and transforms
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **Jupyter**: Interactive notebooks
- **Tqdm**: Progress bars

## 🎯 Key Features

- **Educational Focus**: Each experiment teaches specific concepts
- **Minimal Dependencies**: Only essential libraries required
- **Self-Contained**: No external data dependencies (auto-download)
- **Runnable**: All code works out-of-the-box
- **Well-Documented**: Clear explanations and comments
- **Extensible**: Easy to modify and extend experiments

## 🚀 Getting Started

1. **Environment Setup:**
   ```bash
   # Create virtual environment (recommended)
   python -m venv mini-anns-env
   source mini-anns-env/bin/activate  # On Windows: mini-anns-env\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Run Your First Experiment:**
   ```bash
   jupyter notebook notebooks/01_tiny_image_classifier.ipynb
   ```

3. **Explore All Experiments:**
   ```bash
   jupyter notebook notebooks/
   ```

## 📊 Expected Results

Each notebook produces:
- **Training plots**: Loss curves, accuracy metrics
- **Visualizations**: Predictions, reconstructions, decision boundaries
- **Metrics**: Accuracy, F1-score, reconstruction error, etc.
- **Saved models**: Trained weights for further analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-experiment`)
3. Add your experiment or improvement
4. Ensure all code runs without errors
5. Commit your changes (`git commit -m 'Add amazing experiment'`)
6. Push to the branch (`git push origin feature/amazing-experiment`)
7. Open a Pull Request

## 📝 Adding New Experiments

To add a new experiment:

1. Create a new notebook in `notebooks/` following the naming convention
2. Include:
   - Clear problem statement
   - Dataset loading code
   - Model implementation
   - Training loop
   - Evaluation and visualization
3. Update this README with experiment description
4. Test that it runs without modification

## 🐛 Troubleshooting

**Common Issues:**

- **CUDA errors**: Set `device = torch.device('cpu')` in notebooks
- **Memory issues**: Reduce batch size or model size
- **Import errors**: Ensure all dependencies are installed
- **Dataset download**: Check internet connection for auto-download

**Getting Help:**
- Check the notebook comments for detailed explanations
- Review the `scripts/utils.py` for helper functions
- Open an issue for bugs or questions

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- The machine learning community for educational resources
- Contributors who help improve these experiments

---

**Happy Learning! 🎓**

Start with the first notebook and work your way through. Each experiment builds upon previous concepts while introducing new techniques and applications.
