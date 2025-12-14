#!/usr/bin/env python3
import logging
import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union, Any

class NMTLogger:
    """
    A dedicated logger for the NMT system that handles both console and file logging,
    as well as training metrics visualization.
    """
    
    def __init__(self, log_dir=None, experiment_name=None):
        """
        Initialize the logger with the specified log directory and experiment name.
        
        Args:
            log_dir (str): Directory to store log files
            experiment_name (str): Name of the experiment for log file naming
        """
        if log_dir is None:
            script_dir = Path(__file__).parent.absolute()
            project_dir = script_dir.parent
            log_dir = project_dir / "logs_2k"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up experiment name
        if experiment_name is None:
            experiment_name = f"nmt_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.json"
        
        self.metrics = {
            'train_loss': [],
            'valid_loss': [],
            'bleu': [],
            'learning_rate': [],
            'epochs': [],
            'updates': [],
            'perplexity': [],
            'accuracy': [],
            'cpu_percent': [],
            'memory_percent': [],
            'gpu_load': [],
            'gpu_memory_percent': [],
            'translation_speed': [],
            'sentences_processed': []
        }
        
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Initialized NMT logger for experiment: {experiment_name}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Metrics file: {self.metrics_file}")
    
    def log_info(self, message):
        """Log an informational message."""
        self.logger.info(message)
    
    def log_warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_error(self, message):
        """Log an error message."""
        self.logger.error(message)
    
    def log_metrics(self, metrics_dict, step=None, save=True):
        """
        Log metrics to the metrics dictionary and optionally save to file.
        
        Args:
            metrics_dict (dict): Dictionary of metrics to log
            step (int, optional): Current step. Defaults to None.
            save (bool, optional): Whether to save metrics to file. Defaults to True.
        """
        for k, v in metrics_dict.items():
            if k in self.metrics:
                self.metrics[k].append(v)
        
        if 'step' in metrics_dict and 'epoch' in metrics_dict:
            self.metrics['updates'].append(metrics_dict['step'])
            self.metrics['epochs'].append(metrics_dict['epoch'])
        
        if save:
            self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to a JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_metrics(self, save_dir=None):
        """
        Plot training metrics.
        
        Args:
            save_dir (Path or str, optional): Directory to save plots. Defaults to log_dir.
        """
        if save_dir is None:
            save_dir = self.log_dir
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(exist_ok=True)
        
        if self.metrics['train_loss'] and len(self.metrics['train_loss']) > 0:
            plt.figure(figsize=(10, 6))
            
            x_train = range(len(self.metrics['train_loss']))
            plt.plot(x_train, self.metrics['train_loss'], label='Train Loss')
            
            if self.metrics['valid_loss'] and len(self.metrics['valid_loss']) > 0:
                if len(self.metrics['valid_loss']) < len(self.metrics['train_loss']):
                    valid_interval = len(self.metrics['train_loss']) // len(self.metrics['valid_loss'])
                    x_valid = [i * valid_interval for i in range(len(self.metrics['valid_loss']))]
                    if len(x_valid) > len(self.metrics['valid_loss']):
                        x_valid = x_valid[:len(self.metrics['valid_loss'])]
                else:
                    x_valid = range(len(self.metrics['valid_loss']))
                
                plt.plot(x_valid, self.metrics['valid_loss'], label='Validation Loss')
            
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_dir / f"{self.experiment_name}_loss.png")
            plt.close()
        
        if self.metrics['bleu'] and len(self.metrics['bleu']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(self.metrics['bleu'])), self.metrics['bleu'], label='BLEU Score')
            plt.xlabel('Steps')
            plt.ylabel('BLEU')
            plt.title('BLEU Score')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_dir / f"{self.experiment_name}_bleu.png")
            plt.close()
        
        if self.metrics['learning_rate'] and len(self.metrics['learning_rate']) > 0:
            plt.figure(figsize=(10, 6))
            
            if self.metrics['updates'] and len(self.metrics['updates']) == len(self.metrics['learning_rate']):
                plt.plot(self.metrics['updates'], self.metrics['learning_rate'], label='Learning Rate')
                plt.xlabel('Updates')
            else:
                plt.plot(range(len(self.metrics['learning_rate'])), self.metrics['learning_rate'], label='Learning Rate')
                plt.xlabel('Steps')
                
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_dir / f"{self.experiment_name}_lr.png")
            plt.close()
        
        resource_metrics = ['cpu_percent', 'gpu_load', 'memory_percent']
        has_resource_data = any(self.metrics[metric] and len(self.metrics[metric]) > 0 for metric in resource_metrics)
        
        if has_resource_data:
            plt.figure(figsize=(10, 6))
            
            for metric in resource_metrics:
                if self.metrics[metric] and len(self.metrics[metric]) > 0:
                    plt.plot(range(len(self.metrics[metric])), self.metrics[metric], label=f'{metric.replace("_", " ").title()} (%)')
            
            plt.xlabel('Monitoring Steps')
            plt.ylabel('Usage (%)')
            plt.title('Resource Usage')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_dir / f"{self.experiment_name}_resources.png")
            plt.close()
        
        self.logger.info("Finished plotting metrics")

def main():
    """Example usage of the NMTLogger"""
    logger = NMTLogger()
    
    logger.log_info("Starting NMT training")
    
    for i in range(10):
        metrics = {
            'train_loss': 5.0 - i * 0.4 + np.random.normal(0, 0.1),
            'valid_loss': 5.5 - i * 0.35 + np.random.normal(0, 0.2),
            'bleu': 10 + i * 2 + np.random.normal(0, 0.5),
            'learning_rate': 0.001 * (0.9 ** i),
            'epoch': i // 3,
            'step': i * 100
        }
        logger.log_metrics(metrics, step=i*100)
    
    logger.plot_metrics()
    
    logger.log_info("NMT training completed")

if __name__ == "__main__":
    main() 