import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import *
from model.Garsha import AnomalyDetector
from model.entropy import get_divergence_fn
from data_factory.data_loader import get_loader_segment, SMDSegLoader
from einops import rearrange
from metrics.metrics import *

warnings.filterwarnings('ignore')


class TensorBoardLogger:
    """Enhanced TensorBoard logging with proper initialization and cleanup."""
    
    def __init__(self, log_dir: str = "runs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=self.log_dir / f"experiment_{timestamp}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)
    
    def log_model_graph(self, model: nn.Module, input_tensor: torch.Tensor):
        """Log model graph to TensorBoard."""
        try:
            self.writer.add_graph(model, input_tensor)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


class LearningRateScheduler:
    """Enhanced learning rate scheduling with multiple strategies."""
    
    @staticmethod
    def exponential_decay(optimizer: torch.optim.Optimizer, epoch: int, base_lr: float, decay_rate: float = 0.5):
        """Apply exponential decay to learning rate."""
        lr_adjust = {epoch: base_lr * (decay_rate ** ((epoch - 1) // 1))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return base_lr
    
    @staticmethod
    def cosine_annealing(optimizer: torch.optim.Optimizer, epoch: int, max_epochs: int, base_lr: float, min_lr: float = 1e-6):
        """Apply cosine annealing to learning rate."""
        lr = min_lr + (base_lr - min_lr) * (1 + np.cos(np.pi * epoch / max_epochs)) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class EarlyStopping:
    """Enhanced early stopping with better validation and logging."""
    
    def __init__(self, patience: int = 7, verbose: bool = False, dataset_name: str = '', 
                 delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.restore_best_weights = restore_best_weights
        self.best_model_state = None

    def __call__(self, val_loss: float, val_loss2: float, model: nn.Module, path: str) -> bool:
        """
        Check if early stopping criteria are met.
        
        Returns:
            bool: True if early stopping should be triggered
        """
        score = -val_loss
        score2 = -val_loss2
        
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model_state:
                    model.load_state_dict(self.best_model_state)
                    if self.verbose:
                        print('Restored best model weights')
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0
        
        return self.early_stop

    def save_checkpoint(self, val_loss: float, val_loss2: float, model: nn.Module, path: str):
        """Save model checkpoint and update best scores."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        checkpoint_path = Path(path) / f"{self.dataset}_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        if self.restore_best_weights:
            self.best_model_state = model.state_dict().copy()
        
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class AnomalyMetricsCalculator:
    """Handles anomaly detection metrics and evaluation."""
    
    @staticmethod
    def find_anomaly_starts(ground_truth: np.ndarray) -> np.ndarray:
        """Find indices where anomalies start in ground truth."""
        return np.where((ground_truth[:-1] == 0) & (ground_truth[1:] == 1))[0] + 1
    
    @staticmethod
    def find_detected_anomaly_starts(ground_truth: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Find indices where anomalies are correctly detected."""
        return np.where(
            (ground_truth[:-1] == 0) & (ground_truth[1:] == 1) & 
            (predictions[:-1] == 0) & (predictions[1:] == 1)
        )[0] + 1
    
    @staticmethod
    def apply_point_adjustment(predictions: np.ndarray, ground_truth: np.ndarray, 
                              detected_starts: np.ndarray) -> np.ndarray:
        """Apply point adjustment to fill entire anomaly segments."""
        pred_adjusted = predictions.copy()
        
        for start in detected_starts:
            # Fill backwards to the beginning of the anomaly
            for j in range(start - 1, -1, -1):
                if ground_truth[j] == 0:
                    break
                elif pred_adjusted[j] == 0:
                    pred_adjusted[j] = 1
            
            # Fill forwards to the end of the anomaly
            for j in range(start, len(ground_truth)):
                if ground_truth[j] == 0:
                    break
                elif pred_adjusted[j] == 0:
                    pred_adjusted[j] = 1
        
        return pred_adjusted


class VisualizationManager:
    """Handles plotting and visualization of results."""
    
    @staticmethod
    def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                           save_path: Optional[str] = None):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def smooth_signal(signal: np.ndarray, window_size: int = 6) -> np.ndarray:
        """Apply moving average smoothing to signal."""
        kernel = np.ones(window_size) / window_size
        return np.convolve(signal, kernel, mode='same')


class Solver:
    """Enhanced anomaly detection trainer with improved organization and features."""
    
    DEFAULTS = {
        'patience': 5,
        'temperature': 75,
        'segment_length': 3000,
        'use_tensorboard': True,
        'scheduler_type': 'exponential',  # 'exponential', 'cosine'
        'plot_results': True
    }
    
    def __init__(self, config: Dict[str, Any]):
        # Update configuration with defaults
        self.__dict__.update(self.DEFAULTS, **config)
        
        # Initialize divergence function
        self.divergence_fn = get_divergence_fn(self.divergence, q_param=self.q_param)
        
        # Initialize data loaders
        self._init_data_loaders()
        
        # Build model and setup training
        self._build_model()
        self._setup_training()
        
        # Initialize logging
        if self.use_tensorboard:
            self.tb_logger = TensorBoardLogger()
        
        # Track training history
        self.train_losses = []
        self.val_losses = []

    def _init_data_loaders(self):
        """Initialize all data loaders."""
        base_path = f'dataset/{self.data_path}'
        
        self.train_loader = get_loader_segment(
            self.index, base_path, batch_size=self.batch_size, 
            win_size=self.win_size, mode='train', dataset=self.dataset, shuffle=True
        )
        self.vali_loader = get_loader_segment(
            self.index, base_path, batch_size=self.batch_size,
            win_size=self.win_size, mode='val', dataset=self.dataset
        )
        self.test_loader = get_loader_segment(
            self.index, base_path, batch_size=self.batch_size,
            win_size=self.win_size, mode='test', dataset=self.dataset
        )
        self.thre_loader = get_loader_segment(
            self.index, base_path, batch_size=self.batch_size,
            win_size=self.win_size, mode='thre', dataset=self.dataset
        )

    def _build_model(self):
        """Build and initialize the anomaly detection model."""
        self.model = AnomalyDetector(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            n_heads=self.n_heads,
            d_model=self.d_model,
            e_layers=self.e_layers,
            patch_size=self.patch_size,
            channel=self.input_c
        )
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Model initialized on device: {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def _setup_training(self):
        """Setup training components (optimizer, criterion, etc.)."""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Setup loss criterion
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fuc}")

    def _compute_divergence_loss(self, gpatch: List[torch.Tensor], lpatch: List[torch.Tensor], 
                                temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute divergence loss between global and local patches."""
        gpatch_loss = 0.0
        lpatch_loss = 0.0
        
        for u in range(len(lpatch)):
            # Normalize lpatch
            lpatch_normalized = lpatch[u] / torch.unsqueeze(
                torch.sum(lpatch[u], dim=-1), dim=-1
            ).repeat(1, 1, 1, self.win_size)
            
            if u == 0:
                gpatch_loss = self.divergence_fn(
                    gpatch[u], lpatch_normalized.detach()
                ) * temperature
                lpatch_loss = self.divergence_fn(
                    lpatch_normalized, gpatch[u].detach()
                ) * temperature
            else:
                gpatch_loss += self.divergence_fn(
                    gpatch[u], lpatch_normalized.detach()
                ) * temperature
                lpatch_loss += self.divergence_fn(
                    lpatch_normalized, gpatch[u].detach()
                ) * temperature
        
        return gpatch_loss, lpatch_loss

    def validate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Perform validation on given data loader."""
        self.model.eval()
        loss_1_list = []
        loss_2_list = []
        
        with torch.no_grad():
            for input_data, _ in data_loader:
                input_tensor = input_data.float().to(self.device)
                gpatch, lpatch = self.model(input_tensor)
                
                gpatch_loss, lpatch_loss = self._compute_divergence_loss(
                    gpatch, lpatch, temperature=1.0
                )
                
                # Normalize by number of patches
                gpatch_loss = gpatch_loss / len(lpatch)
                lpatch_loss = lpatch_loss / len(lpatch)
                
                loss_1_list.append((lpatch_loss - gpatch_loss).item())
                loss_2_list.append((lpatch_loss + gpatch_loss).item())
        
        return np.mean(loss_1_list), np.mean(loss_2_list)

    def train(self):
        """Enhanced training loop with better progress tracking and logging."""
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")
        
        # Setup training environment
        start_time = time.time()
        model_save_path = Path(self.model_save_path)
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=True, dataset_name=self.data_path
        )
        
        # Training loop with progress bars
        epoch_pbar = tqdm(range(self.num_epochs), desc="Training Progress")
        
        for epoch in epoch_pbar:
            # Refresh train loader for shuffling
            self.train_loader = get_loader_segment(
                self.index, f'dataset/{self.data_path}',
                batch_size=self.batch_size, win_size=self.win_size,
                mode='train', dataset=self.dataset, shuffle=True
            )
            
            # Training phase
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            train_steps = len(self.train_loader)
            
            batch_pbar = tqdm(
                enumerate(self.train_loader),
                total=train_steps,
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                leave=False
            )
            
            for batch_idx, (input_data, labels) in batch_pbar:
                self.optimizer.zero_grad()
                
                input_tensor = input_data.float().to(self.device)
                gpatch, lpatch = self.model(input_tensor)
                
                # Compute loss
                gpatch_loss, lpatch_loss = self._compute_divergence_loss(
                    gpatch, lpatch, temperature=1.0
                )
                
                gpatch_loss = gpatch_loss / len(lpatch)
                lpatch_loss = lpatch_loss / len(lpatch)
                loss = lpatch_loss - gpatch_loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Calculate epoch metrics
            avg_epoch_loss = running_loss / train_steps
            self.train_losses.append(avg_epoch_loss)
            
            # Validation
            val_loss1, val_loss2 = self.validate(self.vali_loader)
            self.val_losses.append(val_loss1)
            
            # Learning rate scheduling
            if self.scheduler_type == 'exponential':
                current_lr = LearningRateScheduler.exponential_decay(
                    self.optimizer, epoch + 1, self.lr
                )
            elif self.scheduler_type == 'cosine':
                current_lr = LearningRateScheduler.cosine_annealing(
                    self.optimizer, epoch, self.num_epochs, self.lr
                )
            
            # Logging
            if self.use_tensorboard:
                self.tb_logger.log_scalar('Train/Loss', avg_epoch_loss, epoch)
                self.tb_logger.log_scalar('Validation/Loss1', val_loss1, epoch)
                self.tb_logger.log_scalar('Validation/Loss2', val_loss2, epoch)
                self.tb_logger.log_scalar('Learning_Rate', current_lr, epoch)
            
            # Update progress bar
            epoch_pbar.set_postfix({
                "train_loss": f"{avg_epoch_loss:.4f}",
                "val_loss": f"{val_loss1:.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1:3d}: Loss={avg_epoch_loss:.6f}, "
                  f"Val_Loss={val_loss1:.6f}, Time={epoch_time:.2f}s")
            
            # Early stopping check
            if early_stopping(val_loss1, val_loss2, self.model, str(model_save_path)):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Training completion
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED - Total time: {total_time:.2f}s")
        print(f"{'='*60}")
        
        # Plot training curves
        if self.plot_results:
            self.plot_training_history()
        
        # Close TensorBoard logger
        if self.use_tensorboard:
            self.tb_logger.close()

    def test(self) -> Tuple[float, float, float, float]:
        """Enhanced testing with comprehensive evaluation and visualization."""
        print(f"\n{'='*60}")
        print("STARTING EVALUATION")
        print(f"{'='*60}")
        
        # Load best model
        checkpoint_path = Path(self.model_save_path) / f"{self.data_path}_checkpoint.pth"
        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded model from: {checkpoint_path}")
        else:
            print("Warning: No checkpoint found, using current model state")
        
        self.model.eval()
        
        # Phase 1: Collect training statistics
        print("Phase 1: Collecting training statistics...")
        train_energy, train_metric = self._collect_statistics(self.train_loader)
        
        # Phase 2: Determine threshold
        print("Phase 2: Determining threshold...")
        test_energy_thresh, _ = self._collect_statistics(self.thre_loader)
        combined_energy = np.concatenate([train_energy, test_energy_thresh])
        threshold = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print(f"\033[94mThreshold: {threshold:.6f}\033[0m")
        
        # Phase 3: Final evaluation
        print("Phase 3: Final evaluation...")
        test_energy, test_metric, test_labels = self._evaluate_test_set()
        
        # Generate predictions
        predictions = (test_energy > threshold).astype(int)
        ground_truth = test_labels.astype(int)
        
        # Calculate comprehensive metrics
        scores = self._calculate_comprehensive_metrics(predictions, ground_truth, test_energy)
        
        # Apply point adjustment and recalculate metrics
        adjusted_predictions = self._apply_point_adjustment(predictions, ground_truth)
        final_metrics = self._calculate_final_metrics(adjusted_predictions, ground_truth)
        
        # Visualization and analysis
        if self.plot_results:
            self._create_visualizations(
                test_energy, test_metric, ground_truth, predictions, threshold
            )
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETED")
        print(f"{'='*60}")
        
        return final_metrics

    def _collect_statistics(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Collect energy and metric statistics from data loader."""
        energy_list = []
        metric_list = []
        
        with torch.no_grad():
            for input_data, _ in tqdm(data_loader, desc="Collecting statistics", leave=False):
                input_tensor = input_data.float().to(self.device)
                gpatch, lpatch = self.model(input_tensor)
                
                gpatch_loss, lpatch_loss = self._compute_divergence_loss(
                    gpatch, lpatch, temperature=self.temperature
                )
                
                metric = torch.softmax((-gpatch_loss - lpatch_loss), dim=-1)
                
                energy_list.append(metric.detach().cpu().numpy())
                metric_list.append(metric.detach().cpu().numpy())
        
        energy = np.concatenate(energy_list, axis=0).reshape(-1)
        metric = np.concatenate(metric_list, axis=0).reshape(-1)
        
        return energy, metric

    def _evaluate_test_set(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model on test set and return energy, metrics, and labels."""
        energy_list = []
        metric_list = []
        label_list = []
        
        with torch.no_grad():
            for input_data, labels in tqdm(self.thre_loader, desc="Evaluating test set", leave=False):
                input_tensor = input_data.float().to(self.device)
                gpatch, lpatch = self.model(input_tensor)
                
                gpatch_loss, lpatch_loss = self._compute_divergence_loss(
                    gpatch, lpatch, temperature=self.temperature
                )
                
                metric = torch.softmax((-gpatch_loss - lpatch_loss), dim=-1)
                
                energy_list.append(metric.detach().cpu().numpy())
                metric_list.append(metric.detach().cpu().numpy())
                label_list.append(labels.numpy())
        
        energy = np.concatenate(energy_list, axis=0).reshape(-1)
        metric = np.concatenate(metric_list, axis=0).reshape(-1)
        labels = np.concatenate(label_list, axis=0).reshape(-1)
        
        return energy, metric, labels

    def _calculate_comprehensive_metrics(self, predictions: np.ndarray, 
                                       ground_truth: np.ndarray, 
                                       scores: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        print("\n" + "="*50)
        print("COMPREHENSIVE METRICS")
        print("="*50)
        
        with tqdm(total=1, desc="Computing metrics") as pbar:
            metrics = combine_all_evaluation_scores(predictions, ground_truth, scores)
            pbar.update(1)
        
        for key, value in metrics.items():
            print(f'{key:25} : {value:0.4f}')
        
        return metrics

    def _apply_point_adjustment(self, predictions: np.ndarray, 
                               ground_truth: np.ndarray) -> np.ndarray:
        """Apply point adjustment to predictions."""
        print("\n" + "="*50)
        print("POINT ADJUSTMENT ANALYSIS")
        print("="*50)
        
        # Find anomaly starts
        anomaly_starts = AnomalyMetricsCalculator.find_anomaly_starts(ground_truth)
        detected_anomaly_starts = AnomalyMetricsCalculator.find_detected_anomaly_starts(
            ground_truth, predictions
        )
        
        print(f"Ground truth anomaly starts: {len(anomaly_starts)}")
        print(f"Detected anomaly starts: {len(detected_anomaly_starts)}")
        
        if len(detected_anomaly_starts) == 0:
            print("No anomalies were correctly detected by the model.")
            return predictions
        
        print(f"Detected anomaly start indices: {detected_anomaly_starts}")
        
        # Apply point adjustment
        adjusted_predictions = AnomalyMetricsCalculator.apply_point_adjustment(
            predictions, ground_truth, detected_anomaly_starts
        )
        
        return adjusted_predictions

    def _calculate_final_metrics(self, predictions: np.ndarray, 
                                ground_truth: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate final accuracy, precision, recall, and F-score."""
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='binary'
        )
        
        print("\n" + "="*50)
        print("FINAL METRICS")
        print("="*50)
        print(f"Accuracy  : {accuracy:0.4f}")
        print(f"Precision : {precision:0.4f}")
        print(f"Recall    : {recall:0.4f}")
        print(f"F-score   : {f_score:0.4f}")
        
        return accuracy, precision, recall, f_score



# Backward compatibility
def adjust_learning_rate(optimizer, epoch, lr_):
    """Deprecated: Use LearningRateScheduler class instead."""
    return LearningRateScheduler.exponential_decay(optimizer, epoch, lr_)