"""
# Implementation inspired by:
# [1] Xu et al. "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy"
# [2] Yang et al. "DCdetector: Dual Attention Contrastive Representation Learning"  
# [3] Su et al. "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network"
"""  

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from torch.backends import cudnn

from utils.utils import mkdir
from train_factory.trainer import Solver as Trainer


class DualLogger:

    def __init__(self, log_file: str, console_level: int = logging.INFO, file_level: int = logging.DEBUG):
        self.logger = logging.getLogger('transformer_logger')
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)


class ConfigValidator:
     
    @staticmethod
    def validate_config(config: argparse.Namespace) -> None:

        if config.win_size <= 0:
            raise ValueError(f"Window size must be positive, got {config.win_size}")
        
        if config.num_epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, got {config.num_epochs}")
        
        if config.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {config.batch_size}")
        
        if config.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {config.lr}")
        
        if not 0 <= config.anormly_ratio <= 100:
            raise ValueError(f"Anomaly ratio must be between 0 and 100, got {config.anormly_ratio}")
        
        if config.divergence not in ['tsallis', 'renyi', 'kl']:
            raise ValueError(f"Invalid divergence type: {config.divergence}")
    
    @staticmethod
    def process_patch_size(patch_size_input: List) -> List[int]:

        try:
            if isinstance(patch_size_input, list):
                return [int(patch) for patch in patch_size_input]
            else:
                return [int(patch_size_input)]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid patch size format: {patch_size_input}") from e
    
    @staticmethod
    def calculate_optimal_batch_size(dataset: str, data_path: str, index: int, win_size: int) -> int:

        batch_size_options = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        
        dataset_file_mapping = {
            'UCR': f"dataset/{data_path}/UCR_{index}_train.npy",
            'UCR_AUG': f"dataset/{data_path}/UCR_AUG_{index}_train.npy", 
            'SMD_Ori': f"dataset/{data_path}/SMD_Ori_{index}_train.npy"
        }
        
        if dataset not in dataset_file_mapping:
            return 128  # Default batch size
        
        try:
            file_path = dataset_file_mapping[dataset]
            if not os.path.exists(file_path):
                logging.warning(f"Dataset file not found: {file_path}. Using default batch size.")
                return 128
            
            data_len = np.load(file_path).shape[0]
            target_batch_size = data_len / win_size
            
            # Find nearest valid batch size
            return min(batch_size_options, key=lambda x: abs(x - target_batch_size))
            
        except Exception as e:
            logging.warning(f"Error calculating batch size: {e}. Using default.")
            return 128


class GPUManager:

    
    @staticmethod
    def setup_gpu_config(config: argparse.Namespace) -> argparse.Namespace:
        """Setup GPU configuration based on availability and user preferences."""
        config.use_gpu = torch.cuda.is_available() and config.use_gpu
        
        if config.use_gpu:
            if config.use_multi_gpu and torch.cuda.device_count() > 1:
                config.devices = config.devices.replace(' ', '')
                device_ids = config.devices.split(',')
                
                # Validate device IDs
                available_devices = list(range(torch.cuda.device_count()))
                valid_device_ids = []
                
                for device_id in device_ids:
                    try:
                        device_id = int(device_id)
                        if device_id in available_devices:
                            valid_device_ids.append(device_id)
                        else:
                            logging.warning(f"GPU device {device_id} not available. Skipping.")
                    except ValueError:
                        logging.warning(f"Invalid device ID: {device_id}. Skipping.")
                
                if valid_device_ids:
                    config.device_ids = valid_device_ids
                    config.gpu = config.device_ids[0]
                    logging.info(f"Using GPUs: {config.device_ids}")
                else:
                    config.use_multi_gpu = False
                    config.gpu = 0
                    logging.warning("No valid GPU devices found. Falling back to single GPU.")
            else:
                config.gpu = 0
                config.use_multi_gpu = False
                
            logging.info(f"GPU configuration: Device {config.gpu}, Multi-GPU: {config.use_multi_gpu}")
        else:
            logging.info("Using CPU for computation")
            
        return config


def create_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description='Transformer-based Anomaly Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model architecture parameters
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--win_size', type=int, default=100, 
                           help='Input window size')
    model_group.add_argument('--patch_size', type=list, default=[5], 
                           help='Patch sizes for multi-scale analysis')
    model_group.add_argument('--n_heads', type=int, default=1, 
                           help='Number of attention heads')
    model_group.add_argument('--e_layers', type=int, default=3, 
                           help='Number of encoder layers')
    model_group.add_argument('--d_model', type=int, default=256, 
                           help='Model dimension')
    model_group.add_argument('--k', type=int, default=3, 
                           help='Top-k parameter')

    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--lr', type=float, default=1e-4, 
                               help='Learning rate')
    training_group.add_argument('--loss_fuc', type=str, default='MSE',
                               choices=['MSE', 'MAE'], help='Loss function')
    training_group.add_argument('--num_epochs', type=int, default=10, 
                               help='Number of training epochs')
    training_group.add_argument('--batch_size', type=int, default=128, 
                               help='Batch size')

    # Data parameters
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--input_c', type=int, default=9, 
                           help='Number of input channels')
    data_group.add_argument('--output_c', type=int, default=9, 
                           help='Number of output channels')
    data_group.add_argument('--dataset', type=str, default='credit',
                           help='Dataset name')
    data_group.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv',
                           help='Path to dataset')
    data_group.add_argument('--index', type=int, default=137,
                           help='Dataset index')

    # GPU parameters
    gpu_group = parser.add_argument_group('GPU Configuration')
    gpu_group.add_argument('--use_gpu', type=bool, default=True, 
                          help='Use GPU if available')
    gpu_group.add_argument('--gpu', type=int, default=0, 
                          help='GPU device ID')
    gpu_group.add_argument('--use_multi_gpu', action='store_true', default=True,
                          help='Use multiple GPUs')
    gpu_group.add_argument('--devices', type=str, default='0,1,2,3',
                          help='GPU device IDs for multi-GPU training')

    # Anomaly detection parameters
    anomaly_group = parser.add_argument_group('Anomaly Detection')
    anomaly_group.add_argument('--anormly_ratio', type=float, default=4.00,
                              help='Anomaly ratio percentage')
    anomaly_group.add_argument('--divergence', type=str, default='tsallis',
                              choices=['tsallis', 'renyi', 'kl'],
                              help='Divergence measure for anomaly detection')
    anomaly_group.add_argument('--q_param', type=float, default=0.5,
                              help='Q parameter for Tsallis/Renyi divergence')

    # General parameters
    general_group = parser.add_argument_group('General')
    general_group.add_argument('--mode', type=str, default='train',
                              choices=['train', 'test'], help='Operation mode')
    general_group.add_argument('--model_save_path', type=str, default='checkpoints',
                              help='Path to save model checkpoints')
    general_group.add_argument('--rec_timeseries', action='store_true', default=True,
                              help='Reconstruct time series')

    return parser


def setup_environment(config: argparse.Namespace) -> DualLogger:

    # Create necessary directories
    Path(config.model_save_path).mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = f"result/{Path(config.data_path).stem}.log"
    logger = DualLogger(log_file)
    
    # Configure PyTorch
    cudnn.benchmark = True
    
    return logger


def print_configuration(config: argparse.Namespace, logger: DualLogger) -> None:

    config_dict = vars(config)
    
    if config.mode == 'train':
        logger.info("\n" + "="*60)
        logger.info(f"Training Session - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Divergence: {config.divergence}(q={config.q_param}) | Epochs: {config.num_epochs}")
        logger.info("="*60)
        logger.info("HYPERPARAMETERS:")
        logger.info("-" * 40)
        
        # Group parameters for better readability
        param_groups = {
            'Model': ['win_size', 'patch_size', 'n_heads', 'e_layers', 'd_model', 'input_c', 'output_c'],
            'Training': ['lr', 'num_epochs', 'batch_size', 'loss_fuc'],
            'Data': ['dataset', 'data_path', 'index'],
            'Anomaly Detection': ['anormly_ratio', 'divergence', 'q_param'],
            'Hardware': ['use_gpu', 'use_multi_gpu', 'devices']
        }
        
        for group_name, params in param_groups.items():
            logger.info(f"\n{group_name}:")
            for param in params:
                if param in config_dict:
                    logger.info(f"  {param}: {config_dict[param]}")
        
        logger.info("\n" + "="*60)
        logger.info("STARTING TRAINING...")
        logger.info("="*60)
        
    elif config.mode == 'test':
        logger.info("\n" + "="*60)
        logger.info("TEST MODE - Evaluation in Progress")
        logger.info("="*60)


def main(config: argparse.Namespace) -> Optional[Trainer]:

    try:
        # Setup environment and logging
        logger = setup_environment(config)
        
        # Validate and process configuration
        ConfigValidator.validate_config(config)
        config.patch_size = ConfigValidator.process_patch_size(config.patch_size)
        
        # Calculate optimal batch size for specific datasets
        if config.dataset in ['UCR', 'UCR_AUG', 'SMD_Ori']:
            config.batch_size = ConfigValidator.calculate_optimal_batch_size(
                config.dataset, config.data_path, config.index, config.win_size
            )
            logger.info(f"Calculated optimal batch size: {config.batch_size}")
        
        # Setup GPU configuration
        config = GPUManager.setup_gpu_config(config)
        
        # Print configuration
        print_configuration(config, logger)
        
        # Initialize solver
        logger.info("Initializing solver...")
        solver = Trainer(vars(config))
        
        # Execute based on mode
        if config.mode == 'train':
            logger.info("Starting training process...")
            solver.train()
            logger.info("Training completed successfully!")
            
        elif config.mode == 'test':
            logger.info("Starting evaluation process...")
            results = solver.test()
            logger.info("Evaluation completed successfully!")
            
            if results:
                accuracy, precision, recall, f_score = results
                logger.info(f"Final Results - Accuracy: {accuracy:.4f}, "
                          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")
        
        return solver
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error in main execution: {str(e)}")
        else:
            print(f"Error in main execution: {str(e)}")
        raise


if __name__ == '__main__':
    try:
        # Parse arguments
        parser = create_argument_parser()
        config = parser.parse_args()
        
        # Run main function
        solver = main(config)
        
        if solver:
            print(f"\n{'='*60}")
            print("EXECUTION COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)
