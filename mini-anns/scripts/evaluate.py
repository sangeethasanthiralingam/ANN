"""
Evaluation utilities for Mini-ANNs project.
Provides comprehensive evaluation functions for different types of neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score,
                           mean_squared_error, mean_absolute_error, r2_score)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from .utils import get_device, plot_confusion_matrix, ensure_dir

class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model, device=None, save_dir="results/plots"):
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.save_dir = save_dir
        ensure_dir(save_dir)
        
    def evaluate_classification(self, data_loader, class_names=None, verbose=True):
        """Evaluate classification model."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        avg_loss = total_loss / len(data_loader)
        
        # Calculate AUC for binary classification
        auc = None
        if len(np.unique(all_labels)) == 2:
            try:
                auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
            except:
                auc = None
        
        if verbose:
            print(f"Classification Results:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Avg Loss:  {avg_loss:.4f}")
            if auc is not None:
                print(f"  AUC:       {auc:.4f}")
        
        # Plot confusion matrix
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(all_labels)))]
        
        plot_confusion_matrix(all_labels, all_preds, class_names, 
                            title="Confusion Matrix", 
                            save_path=os.path.join(self.save_dir, "confusion_matrix.png"))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'avg_loss': avg_loss,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def evaluate_regression(self, data_loader, verbose=True):
        """Evaluate regression model."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                all_preds.extend(output.cpu().numpy().flatten())
                all_labels.extend(target.cpu().numpy().flatten())
        
        # Calculate metrics
        mse = mean_squared_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        avg_loss = total_loss / len(data_loader)
        
        if verbose:
            print(f"Regression Results:")
            print(f"  MSE:       {mse:.6f}")
            print(f"  RMSE:      {rmse:.6f}")
            print(f"  MAE:       {mae:.6f}")
            print(f"  R² Score:  {r2:.4f}")
            print(f"  Avg Loss:  {avg_loss:.6f}")
        
        # Plot predictions vs actual
        self._plot_regression_results(all_labels, all_preds)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'avg_loss': avg_loss,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def _plot_regression_results(self, y_true, y_pred, save_path=None):
        """Plot regression results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot: predicted vs actual
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title('Predicted vs Actual')
        axes[0].grid(True)
        
        # Residuals plot
        residuals = np.array(y_pred) - np.array(y_true)
        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')
        axes[1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, "regression_results.png"), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names=None, top_k=10, save_path=None):
        """Plot feature importance (for models that support it)."""
        # This is a placeholder - would need to be implemented based on model type
        print("Feature importance plotting not implemented for this model type")
    
    def plot_latent_space(self, data_loader, method='tsne', n_components=2, 
                         class_names=None, save_path=None):
        """Plot latent space representation (for autoencoders, etc.)."""
        self.model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                
                # Get features from the model
                if hasattr(self.model, 'encoder'):
                    # Autoencoder
                    features_batch = self.model.encoder(data)
                elif hasattr(self.model, 'features'):
                    # CNN with features method
                    features_batch = self.model.features(data)
                else:
                    # Use last layer before classification
                    features_batch = data.view(data.size(0), -1)
                    for layer in self.model.children():
                        if isinstance(layer, nn.Linear) and layer.out_features > 2:
                            features_batch = layer(features_batch)
                        elif isinstance(layer, nn.Linear) and layer.out_features <= 2:
                            break
                
                features.extend(features_batch.cpu().numpy())
                labels.extend(target.numpy())
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=n_components)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        features_reduced = reducer.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], 
                            c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'Latent Space ({method.upper()})')
        plt.xlabel(f'Component 1')
        plt.ylabel(f'Component 2')
        
        if class_names:
            plt.legend(class_names)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, f"latent_space_{method}.png"), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_curves(self, train_losses, val_losses=None, 
                           train_accs=None, val_accs=None, save_path=None):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2 if val_accs is not None else 1, figsize=(12, 4))
        if val_accs is None:
            axes = [axes]
        
        # Loss curves
        axes[0].plot(train_losses, label='Train Loss', color='blue')
        if val_losses is not None:
            axes[0].plot(val_losses, label='Validation Loss', color='red')
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curves
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
        else:
            plt.savefig(os.path.join(self.save_dir, "training_curves.png"), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results, model_name="Model", save_path=None):
        """Generate a comprehensive evaluation report."""
        report = f"""
# {model_name} Evaluation Report

## Model Information
- Model: {self.model.__class__.__name__}
- Parameters: {sum(p.numel() for p in self.model.parameters()):,}
- Device: {self.device}

## Results
"""
        
        if 'accuracy' in results:
            # Classification results
            report += f"""
### Classification Metrics
- Accuracy:  {results['accuracy']:.4f}
- Precision: {results['precision']:.4f}
- Recall:    {results['recall']:.4f}
- F1-Score:  {results['f1_score']:.4f}
- Avg Loss:  {results['avg_loss']:.4f}
"""
            if results.get('auc') is not None:
                report += f"- AUC:       {results['auc']:.4f}\n"
        
        elif 'mse' in results:
            # Regression results
            report += f"""
### Regression Metrics
- MSE:       {results['mse']:.6f}
- RMSE:      {results['rmse']:.6f}
- MAE:       {results['mae']:.6f}
- R² Score:  {results['r2_score']:.4f}
- Avg Loss:  {results['avg_loss']:.6f}
"""
        
        report += f"""
## Files Generated
- Confusion Matrix: confusion_matrix.png
- Training Curves: training_curves.png
- Latent Space: latent_space_tsne.png
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        else:
            with open(os.path.join(self.save_dir, "evaluation_report.md"), 'w') as f:
                f.write(report)
        
        print(report)

def evaluate_model(model, data_loader, task_type='classification', device=None, 
                  save_dir="results/plots", **kwargs):
    """
    Convenience function to evaluate a model.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        task_type: 'classification' or 'regression'
        device: Device to use for evaluation
        save_dir: Directory to save plots and reports
        **kwargs: Additional arguments for evaluator
    
    Returns:
        Dictionary with evaluation results
    """
    evaluator = ModelEvaluator(model, device, save_dir)
    
    if task_type == 'classification':
        return evaluator.evaluate_classification(data_loader, **kwargs)
    elif task_type == 'regression':
        return evaluator.evaluate_regression(data_loader, **kwargs)
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")

if __name__ == "__main__":
    # Example usage
    print("This is a utility module. Import and use the ModelEvaluator class or evaluate_model function.")
    print("Example:")
    print("from scripts.evaluate import evaluate_model")
    print("results = evaluate_model(model, test_loader, 'classification')")
