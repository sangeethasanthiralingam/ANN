"""
Generic training script for Mini-ANNs project.
Provides reusable training loops for classification and regression tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from .utils import plot_training_history, save_model, get_device, set_seed

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()

class Trainer:
    """Generic trainer class for neural networks."""
    
    def __init__(self, model, device=None, save_dir="results/logs"):
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if output.dim() > 1:  # Classification
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                if output.dim() > 1:  # Classification
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def train_classification(self, train_loader, val_loader=None, epochs=10, lr=0.001, 
                           weight_decay=1e-4, scheduler=None, early_stopping=None, 
                           save_best=True, verbose=True):
        """Train model for classification task."""
        
        # Setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if scheduler is None:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//2, gamma=0.1)
        
        if early_stopping is None:
            early_stopping = EarlyStopping(patience=10)
        
        best_val_acc = 0
        start_time = time.time()
        
        if verbose:
            print(f"Starting training for {epochs} epochs...")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Device: {self.device}")
            print("-" * 50)
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.validate_epoch(val_loader, criterion)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                # Save best model
                if save_best and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_model(self.model, os.path.join(self.save_dir, 'best_model.pth'), 
                             optimizer, epoch, val_loss)
                
                # Early stopping
                if early_stopping(val_loss, self.model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                val_loss, val_acc = 0, 0
            
            # Learning rate scheduling
            scheduler.step()
            
            # Logging
            if verbose and (epoch + 1) % 1 == 0:
                elapsed = time.time() - start_time
                if val_loader is not None:
                    print(f"Epoch {epoch+1:3d}/{epochs}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                          f"Time: {elapsed:.1f}s")
                else:
                    print(f"Epoch {epoch+1:3d}/{epochs}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                          f"Time: {elapsed:.1f}s")
        
        # Final model save
        save_model(self.model, os.path.join(self.save_dir, 'final_model.pth'), 
                  optimizer, epoch, val_loss if val_loader else train_loss)
        
        if verbose:
            print(f"\nTraining completed in {time.time() - start_time:.1f}s")
            if val_loader is not None:
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def train_regression(self, train_loader, val_loader=None, epochs=10, lr=0.001, 
                        weight_decay=1e-4, scheduler=None, early_stopping=None, 
                        save_best=True, verbose=True):
        """Train model for regression task."""
        
        # Setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if scheduler is None:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//2, gamma=0.1)
        
        if early_stopping is None:
            early_stopping = EarlyStopping(patience=10)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        if verbose:
            print(f"Starting regression training for {epochs} epochs...")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Device: {self.device}")
            print("-" * 50)
        
        for epoch in range(epochs):
            # Training
            train_loss, _ = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss, _ = self.validate_epoch(val_loader, criterion)
                self.val_losses.append(val_loss)
                
                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(self.model, os.path.join(self.save_dir, 'best_model.pth'), 
                             optimizer, epoch, val_loss)
                
                # Early stopping
                if early_stopping(val_loss, self.model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                val_loss = 0
            
            # Learning rate scheduling
            scheduler.step()
            
            # Logging
            if verbose and (epoch + 1) % 1 == 0:
                elapsed = time.time() - start_time
                if val_loader is not None:
                    print(f"Epoch {epoch+1:3d}/{epochs}: "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | "
                          f"Time: {elapsed:.1f}s")
                else:
                    print(f"Epoch {epoch+1:3d}/{epochs}: "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Time: {elapsed:.1f}s")
        
        # Final model save
        save_model(self.model, os.path.join(self.save_dir, 'final_model.pth'), 
                  optimizer, epoch, val_loss if val_loader else train_loss)
        
        if verbose:
            print(f"\nTraining completed in {time.time() - start_time:.1f}s")
            if val_loader is not None:
                print(f"Best validation loss: {best_val_loss:.6f}")
        
        return self.train_losses, self.val_losses
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        plot_training_history(
            self.train_losses, 
            self.val_losses if self.val_losses else None,
            self.train_accuracies if self.train_accuracies else None,
            self.val_accuracies if self.val_accuracies else None,
            save_path=save_path
        )

def train_model(model, train_loader, val_loader=None, task_type='classification', 
                epochs=10, lr=0.001, device=None, save_dir="results/logs", **kwargs):
    """
    Convenience function to train a model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        task_type: 'classification' or 'regression'
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to use for training
        save_dir: Directory to save models and logs
        **kwargs: Additional arguments for trainer
    
    Returns:
        Trainer object with training history
    """
    trainer = Trainer(model, device, save_dir)
    
    if task_type == 'classification':
        trainer.train_classification(train_loader, val_loader, epochs, lr, **kwargs)
    elif task_type == 'regression':
        trainer.train_regression(train_loader, val_loader, epochs, lr, **kwargs)
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")
    
    return trainer

if __name__ == "__main__":
    # Example usage
    print("This is a utility module. Import and use the Trainer class or train_model function.")
    print("Example:")
    print("from scripts.train import train_model")
    print("trainer = train_model(model, train_loader, val_loader, 'classification')")
