#!/usr/bin/env python3
"""
Comprehensive logging module for NMT experiments.
This module provides functions to log training progress, model configurations,
evaluation results, and system information.
"""

import os
import sys
import json
import yaml
import time
import logging
import platform
import subprocess
import shutil
from datetime import datetime
import socket
import uuid
import psutil
try:
    import torch
    import numpy as np
except ImportError:
    pass  # Handle gracefully if not available

class NMTLogger:
    """
    NMT Logger class for comprehensive experiment tracking.
    """
    
    def __init__(self, experiment_name=None, base_dir=None, config=None):
        """
        Initialize the logger.
        
        Args:
            experiment_name (str, optional): Name of the experiment. If None, a timestamp will be used.
            base_dir (str, optional): Base directory for logs. If None, '../logs' will be used.
            config (dict, optional): Configuration dictionary to log.
        """
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = str(uuid.uuid4())[:8]
            self.experiment_name = f"nmt_experiment_{timestamp}_{experiment_id}"
        else:
            self.experiment_name = experiment_name
            
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            self.base_dir = os.path.join(project_root, 'logs')
        else:
            self.base_dir = base_dir
            
        self.exp_dir = os.path.join(self.base_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.config_dir = os.path.join(self.exp_dir, 'config')
        self.model_dir = os.path.join(self.exp_dir, 'models')
        self.results_dir = os.path.join(self.exp_dir, 'results')
        self.tensorboard_dir = os.path.join(self.exp_dir, 'tensorboard')
        self.translations_dir = os.path.join(self.exp_dir, 'translations')
        
        for directory in [self.config_dir, self.model_dir, self.results_dir, 
                         self.tensorboard_dir, self.translations_dir]:
            os.makedirs(directory, exist_ok=True)
            
        self.log_file = os.path.join(self.exp_dir, 'experiment.log')
        self.setup_logger()
        
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Log directory: {self.exp_dir}")
        
        self.log_system_info()
        
        if config:
            self.log_config(config)
            
        self.logger.info("Logger initialized successfully")
        
    def setup_logger(self):
        """Set up the Python logger."""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_system_info(self):
        """Log system information."""
        self.logger.info("=== System Information ===")
        
        system_info = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            memory = psutil.virtual_memory()
            system_info["total_memory_gb"] = round(memory.total / (1024**3), 2)
            system_info["available_memory_gb"] = round(memory.available / (1024**3), 2)
        except:
            self.logger.warning("Could not get memory information")
        
        try:
            if 'torch' in sys.modules:
                system_info["cuda_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    system_info["cuda_version"] = torch.version.cuda
                    system_info["gpu_count"] = torch.cuda.device_count()
                    system_info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except:
            self.logger.warning("Could not get GPU information")
        
        for key, value in system_info.items():
            self.logger.info(f"{key}: {value}")
        
        system_info_file = os.path.join(self.exp_dir, 'system_info.json')
        with open(system_info_file, 'w') as f:
            json.dump(system_info, f, indent=2)
    
    def log_config(self, config, filename='config.yaml'):
        """
        Log configuration.
        
        Args:
            config (dict): Configuration dictionary.
            filename (str, optional): Filename to save the configuration.
        """
        self.logger.info("Logging configuration")
        
        config_file = os.path.join(self.config_dir, filename)
        
        if filename.endswith('.json'):
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"Configuration saved to {config_file}")
        
        if isinstance(config, str) and os.path.isfile(config):
            shutil.copy2(config, os.path.join(self.config_dir, os.path.basename(config)))
    
    def log_command(self, command, output=None):
        """
        Log a command and its output.
        
        Args:
            command (str or list): Command that was executed.
            output (str, optional): Output of the command.
        """
        if isinstance(command, list):
            command_str = ' '.join(command)
        else:
            command_str = command
            
        self.logger.info(f"Executing command: {command_str}")
        
        commands_file = os.path.join(self.exp_dir, 'commands.log')
        with open(commands_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {command_str}\n")
            
            if output:
                f.write("Output:\n")
                f.write(output)
                f.write("\n" + "-" * 80 + "\n")
    
    def capture_stdout(self, func, *args, **kwargs):
        """
        Capture stdout from a function and log it.
        
        Args:
            func: Function to execute.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            The return value of the function.
        """
        stdout_file = os.path.join(self.exp_dir, 'stdout.log')
        with open(stdout_file, 'a') as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running {func.__name__}\n")
            
            original_stdout = sys.stdout
            sys.stdout = f
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                sys.stdout = original_stdout
    
    def log_training_progress(self, step, metrics, filename='training_progress.jsonl'):
        """
        Log training progress.
        
        Args:
            step (int): Current training step.
            metrics (dict): Dictionary of metrics.
            filename (str, optional): Filename to save the progress.
        """
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'step': step,
            **metrics
        }
        
        progress_file = os.path.join(self.results_dir, filename)
        with open(progress_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        metrics_str = ', '.join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")
    
    def log_evaluation_results(self, results, filename='evaluation_results.json'):
        """
        Log evaluation results.
        
        Args:
            results (dict): Dictionary of evaluation results.
            filename (str, optional): Filename to save the results.
        """
        self.logger.info("Logging evaluation results")
        
        results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        results_file = os.path.join(self.results_dir, filename)
        
        if os.path.isfile(results_file):
            with open(results_file, 'r') as f:
                try:
                    existing_results = json.load(f)
                    if not isinstance(existing_results, list):
                        existing_results = [existing_results]
                except:
                    existing_results = []
            
            existing_results.append(results)
            
            with open(results_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
        else:
            with open(results_file, 'w') as f:
                json.dump([results], f, indent=2)
        
        self.logger.info("Evaluation results:")
        for key, value in results.items():
            if key != 'timestamp':
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
    
    def log_translation_examples(self, sources, references, hypotheses, indices=None, filename='translation_examples.txt'):
        """
        Log translation examples.
        
        Args:
            sources (list): List of source sentences.
            references (list): List of reference translations.
            hypotheses (list): List of model translations.
            indices (list, optional): List of indices to log. If None, all examples are logged.
            filename (str, optional): Filename to save the examples.
        """
        self.logger.info("Logging translation examples")
        
        if indices is None:
            indices = range(len(sources))
        
        examples_file = os.path.join(self.translations_dir, filename)
        with open(examples_file, 'w') as f:
            f.write(f"Translation Examples - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i in indices:
                if i < len(sources) and i < len(references) and i < len(hypotheses):
                    f.write(f"Example {i+1}:\n")
                    f.write(f"Source: {sources[i]}\n")
                    f.write(f"Reference: {references[i]}\n")
                    f.write(f"Hypothesis: {hypotheses[i]}\n")
                    f.write("\n")
        
        self.logger.info(f"Translation examples saved to {examples_file}")
    
    def save_model_checkpoint(self, model_path, step, metrics=None):
        """
        Save a copy of the model checkpoint.
        
        Args:
            model_path (str): Path to the model checkpoint.
            step (int): Current training step.
            metrics (dict, optional): Dictionary of metrics.
        """
        if not os.path.isfile(model_path):
            self.logger.warning(f"Model file not found: {model_path}")
            return
        
        checkpoint_dir = os.path.join(self.model_dir, f"checkpoint_{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model_filename = os.path.basename(model_path)
        checkpoint_path = os.path.join(checkpoint_dir, model_filename)
        shutil.copy2(model_path, checkpoint_path)
        
        if metrics:
            metrics_file = os.path.join(checkpoint_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Model checkpoint saved to {checkpoint_path}")
    
    def log_tensorboard_dir(self, tensorboard_dir):
        """
        Log the tensorboard directory.
        
        Args:
            tensorboard_dir (str): Path to the tensorboard directory.
        """
        if not os.path.isdir(tensorboard_dir):
            self.logger.warning(f"Tensorboard directory not found: {tensorboard_dir}")
            return
        
        symlink_path = os.path.join(self.tensorboard_dir, 'current')
        
        if os.path.islink(symlink_path):
            os.unlink(symlink_path)
        
        os.symlink(tensorboard_dir, symlink_path, target_is_directory=True)
        
        self.logger.info(f"Tensorboard directory linked: {tensorboard_dir}")
    
    def log_exception(self, exception):
        """
        Log an exception.
        
        Args:
            exception (Exception): Exception to log.
        """
        self.logger.error(f"Exception: {type(exception).__name__}: {str(exception)}")
        
        import traceback
        tb = traceback.format_exc()
        self.logger.error(f"Traceback:\n{tb}")
        
        exceptions_file = os.path.join(self.exp_dir, 'exceptions.log')
        with open(exceptions_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {type(exception).__name__}: {str(exception)}\n")
            f.write(f"Traceback:\n{tb}\n")
            f.write("-" * 80 + "\n")
    
    def finalize(self, status="completed", message=None):
        """
        Finalize the experiment.
        
        Args:
            status (str, optional): Status of the experiment.
            message (str, optional): Additional message.
        """
        self.logger.info(f"Experiment {status}")
        if message:
            self.logger.info(message)
        
        status_file = os.path.join(self.exp_dir, 'status.json')
        status_data = {
            'status': status,
            'message': message,
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        self.create_summary()
        
        self.logger.info(f"Experiment logs saved to {self.exp_dir}")
    
    def create_summary(self):
        """Create a summary of the experiment."""
        summary_file = os.path.join(self.exp_dir, 'summary.md')
        
        with open(summary_file, 'w') as f:
            f.write(f"# Experiment Summary: {self.experiment_name}\n\n")
            
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            system_info_file = os.path.join(self.exp_dir, 'system_info.json')
            if os.path.isfile(system_info_file):
                f.write("## System Information\n\n")
                with open(system_info_file, 'r') as sf:
                    system_info = json.load(sf)
                    for key, value in system_info.items():
                        f.write(f"- **{key}:** {value}\n")
                f.write("\n")
            
            config_files = [f for f in os.listdir(self.config_dir) if f.endswith(('.yaml', '.json'))]
            if config_files:
                f.write("## Configuration\n\n")
                f.write(f"Configuration files: {', '.join(config_files)}\n\n")
                
                if config_files:
                    config_file = os.path.join(self.config_dir, config_files[0])
                    f.write("```yaml\n")
                    with open(config_file, 'r') as cf:
                        f.write(cf.read())
                    f.write("```\n\n")
            
            results_file = os.path.join(self.results_dir, 'evaluation_results.json')
            if os.path.isfile(results_file):
                f.write("## Evaluation Results\n\n")
                with open(results_file, 'r') as rf:
                    results = json.load(rf)
                    if isinstance(results, list):
                        f.write("| Metric | Value | Timestamp |\n")
                        f.write("|--------|-------|----------|\n")
                        
                        for result in results:
                            timestamp = result.get('timestamp', '')
                            for key, value in result.items():
                                if key != 'timestamp':
                                    if isinstance(value, float):
                                        f.write(f"| {key} | {value:.4f} | {timestamp} |\n")
                                    else:
                                        f.write(f"| {key} | {value} | {timestamp} |\n")
                    else:
                        timestamp = results.get('timestamp', '')
                        f.write(f"**Timestamp:** {timestamp}\n\n")
                        
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        
                        for key, value in results.items():
                            if key != 'timestamp':
                                if isinstance(value, float):
                                    f.write(f"| {key} | {value:.4f} |\n")
                                else:
                                    f.write(f"| {key} | {value} |\n")
                f.write("\n")
            
            examples_files = [f for f in os.listdir(self.translations_dir) if f.endswith('.txt')]
            if examples_files:
                f.write("## Translation Examples\n\n")
                
                examples_file = os.path.join(self.translations_dir, examples_files[0])
                with open(examples_file, 'r') as ef:
                    lines = ef.readlines()
                    example_count = 0
                    i = 0
                    while i < len(lines) and example_count < 3:
                        line = lines[i]
                        if line.startswith("Example "):
                            example_count += 1
                            f.write(f"### {line}")
                            for j in range(1, 4):
                                if i + j < len(lines):
                                    f.write(f"{lines[i+j]}")
                            f.write("\n")
                            i += 4
                        else:
                            i += 1
                f.write("\n")
            
            status_file = os.path.join(self.exp_dir, 'status.json')
            if os.path.isfile(status_file):
                f.write("## Status\n\n")
                with open(status_file, 'r') as sf:
                    status = json.load(sf)
                    f.write(f"**Status:** {status.get('status', 'unknown')}\n")
                    if status.get('message'):
                        f.write(f"**Message:** {status.get('message')}\n")
                    f.write(f"**End Time:** {status.get('end_time', '')}\n")
                f.write("\n")
            
            f.write("## Links\n\n")
            f.write(f"- [Log File](experiment.log)\n")
            f.write(f"- [Commands](commands.log)\n")
            if os.path.isdir(os.path.join(self.tensorboard_dir, 'current')):
                f.write(f"- [TensorBoard](tensorboard/current)\n")
            f.write("\n")

# Helper functions for easy usage

def setup_experiment(config_path, experiment_name=None):
    """
    Set up an experiment with the given configuration.
    
    Args:
        config_path (str): Path to the configuration file.
        experiment_name (str, optional): Name of the experiment.
        
        Returns:
            NMTLogger: Logger instance.
        """
        if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
    
    logger = NMTLogger(experiment_name=experiment_name, config=config)
    logger.log_config(config_path)
    
    return logger

def log_training_run(config_path, train_script, args=None, experiment_name=None):
    """
    Log a training run.
    
    Args:
        config_path (str): Path to the configuration file.
        train_script (str): Path to the training script.
        args (list, optional): Additional arguments for the training script.
        experiment_name (str, optional): Name of the experiment.
        
        Returns:
            NMTLogger: Logger instance.
        """
        logger = setup_experiment(config_path, experiment_name)
        
        command = ['python', train_script]
        if args:
            command.extend(args)
        
        logger.log_command(command)
        
        try:
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        output = []
        for line in process.stdout:
            print(line, end='')
            output.append(line)
        
        process.wait()
        
        logger.log_command(command, ''.join(output))
        
        if process.returncode != 0:
            logger.logger.error(f"Command failed with return code {process.returncode}")
            logger.finalize(status="failed", message=f"Command failed with return code {process.returncode}")
        else:
            logger.finalize(status="completed")
        
    except Exception as e:
        logger.log_exception(e)
        logger.finalize(status="failed", message=str(e))
    
    return logger

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NMT Logger')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--train_script', type=str, required=True, help='Path to training script')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('--args', nargs=argparse.REMAINDER, help='Additional arguments for the training script')
    
    args = parser.parse_args()
    
    log_training_run(args.config, args.train_script, args.args, args.experiment_name) 