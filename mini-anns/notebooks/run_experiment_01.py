#!/usr/bin/env python3
"""
Mini-ANNs Experiment 01: Tiny Image Classifier
Run this script to execute the first experiment directly
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("🚀 Starting Mini-ANNs Experiment 01: Tiny Image Classifier")
print("=" * 60)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Using device: {device}")

# Define the model
class TinyImageClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(TinyImageClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load data
print("📊 Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root='../data/mnist', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='../data/mnist', 
    train=False, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Test samples: {len(test_dataset)}")

# Initialize model
model = TinyImageClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training function
def train_model(model, train_loader, test_loader, epochs=5):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        
        # Testing
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_accuracy = 100. * test_correct / test_total
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
    
    return train_losses, train_accuracies, test_accuracies

# Train the model
print("\n🎯 Starting training...")
train_losses, train_accuracies, test_accuracies = train_model(model, train_loader, test_loader, epochs=3)

# Plot results
print("\n📈 Creating visualizations...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot training loss
ax1.plot(train_losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

# Plot accuracies
ax2.plot(train_accuracies, label='Train Accuracy')
ax2.plot(test_accuracies, label='Test Accuracy')
ax2.set_title('Model Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('../results/plots/experiment_01_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Test on a few samples
print("\n🔍 Testing on sample images...")
model.eval()
with torch.no_grad():
    data, target = next(iter(test_loader))
    data, target = data[:8].to(device), target[:8]
    output = model(data)
    _, predicted = output.max(1)
    
    # Display results
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        row, col = i // 4, i % 4
        axes[row, col].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[row, col].set_title(f'True: {target[i].item()}, Pred: {predicted[i].item()}')
        axes[row, col].axis('off')
    
    plt.suptitle('Sample Predictions')
    plt.tight_layout()
    plt.savefig('../results/plots/experiment_01_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

print(f"\n🎉 Experiment 01 Complete!")
print(f"📊 Final Test Accuracy: {test_accuracies[-1]:.2f}%")
print(f"💾 Results saved to: ../results/plots/")
print("=" * 60)
