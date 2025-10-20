"""
Utility functions for Mini-ANNs project.
Contains reusable helper functions for data loading, visualization, and model utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get the best available device (CUDA if available, otherwise CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def load_mnist_data(batch_size=64, test_split=0.2):
    """Load MNIST dataset with train/test split."""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load full dataset
    full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    # Split into train and validation
    train_size = int((1 - test_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Load test set
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def load_cifar10_data(batch_size=64, test_split=0.2):
    """Load CIFAR-10 dataset with train/test split."""
    from torchvision import datasets, transforms
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load full dataset
    full_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    
    # Split into train and validation
    train_size = int((1 - test_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Load test set
    test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def plot_training_history(train_losses, val_losses=None, train_accs=None, val_accs=None, 
                         title="Training History", save_path=None):
    """Plot training history with loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2 if val_accs is not None else 1, figsize=(12, 4))
    if val_accs is None:
        axes = [axes]
    
    # Plot losses
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    if val_losses is not None:
        axes[0].plot(val_losses, label='Validation Loss', color='red')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracies
    if train_accs is not None:
        axes[1].plot(train_accs, label='Train Accuracy', color='blue')
        if val_accs is not None:
            axes[1].plot(val_accs, label='Validation Accuracy', color='red')
        axes[1].set_title('Accuracy Curves')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix with optional class names."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_classification(model, data_loader, device, class_names=None):
    """Evaluate classification model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    if class_names:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return accuracy, avg_loss, all_preds, all_labels

def save_model(model, path, optimizer=None, epoch=None, loss=None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_model(model, path, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {path}")
    return checkpoint.get('epoch', 0), checkpoint.get('loss', None)

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_synthetic_data(n_samples=1000, n_features=2, n_classes=2, noise=0.1, random_state=42):
    """Create synthetic classification dataset."""
    from sklearn.datasets import make_classification, make_circles, make_moons
    
    if n_classes == 2:
        if random_state == 1:  # XOR problem
            X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            y = np.array([0, 1, 1, 0])
            # Duplicate to get more samples
            X = np.tile(X, (n_samples // 4, 1))
            y = np.tile(y, n_samples // 4)
        elif random_state == 2:  # Circles
            X, y = make_circles(n_samples=n_samples, noise=noise, random_state=42)
        else:  # Moons
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    else:
        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                 n_classes=n_classes, n_redundant=0, 
                                 n_informative=n_features, random_state=random_state)
    
    return X, y

def create_time_series_data(n_samples=1000, n_features=1, noise=0.1, trend=0.01, seasonality=True):
    """Create synthetic time series data."""
    t = np.linspace(0, 4 * np.pi, n_samples)
    
    if n_features == 1:
        # Single sine wave with trend and noise
        y = np.sin(t) + trend * t + np.random.normal(0, noise, n_samples)
        if seasonality:
            y += 0.5 * np.sin(2 * t)  # Additional seasonal component
    else:
        # Multiple features
        y = np.zeros((n_samples, n_features))
        for i in range(n_features):
            freq = 1 + i * 0.5  # Different frequencies
            y[:, i] = np.sin(freq * t) + trend * t + np.random.normal(0, noise, n_samples)
    
    return y

def plot_decision_boundary(model, X, y, title="Decision Boundary", save_path=None):
    """Plot decision boundary for 2D classification problems."""
    if X.shape[1] != 2:
        print("Decision boundary plotting only works for 2D data")
        return
    
    # Create a mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get predictions for mesh points
    device = next(model.parameters()).device
    mesh_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    
    model.eval()
    with torch.no_grad():
        Z = model(mesh_points)
        if Z.dim() > 1:
            Z = Z.argmax(dim=1)
        Z = Z.cpu().numpy().reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def ensure_dir(path):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)

def get_model_summary(model, input_size):
    """Print model summary with parameter count."""
    total_params = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error in forward pass: {e}")

# Set default seed
set_seed(42)
