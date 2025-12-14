#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys
import json
import re
import yaml
import time
import psutil
import logging
import glob
try:
    import torch
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
from nmt_logger import NMTLogger

logging.getLogger("onmt.inputters.inputter").setLevel(logging.WARNING)
logging.getLogger("onmt.inputters.corpus").setLevel(logging.WARNING)
logging.getLogger("onmt.inputters.dataset_base").setLevel(logging.WARNING)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Train NMT model using OpenNMT-py')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--log_dir', type=str, default=os.path.join(project_root, 'logs_rnn_small_baseline_tur_1200_setimes'),
                        help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str, default='onmt_training',
                        help='Name of the experiment for logging')
    parser.add_argument('--monitor_resources', action='store_true',
                        help='Monitor system resources during training')
    parser.add_argument('--monitor_interval', type=int, default=60,
                        help='Interval in seconds for resource monitoring')
    parser.add_argument('--log_level', type=str, default='WARNING',
                        help='Log level for the training command')
    parser.add_argument('--train_from', type=str, default=None,
                        help='Path to a checkpoint to continue training from')
    parser.add_argument('--continue_from_last', action='store_true',
                        help='Continue training from the last checkpoint in the models directory')
    parser.add_argument('--config', type=str, default=os.path.join(project_root, 'configs', 'config_bpe_rnn_small_multi_baseline_1200_setimes.yaml'),
                        help='Path to the configuration file')
    return parser.parse_args()

def parse_onmt_output(output_line):
    """Parse OpenNMT-py output line to extract metrics."""
    metrics = {}
    
    step_match = re.search(r'Step (\d+)', output_line)
    if step_match:
        metrics['step'] = int(step_match.group(1))
    
    epoch_match = re.search(r'Epoch (\d+)', output_line)
    if epoch_match:
        metrics['epoch'] = int(epoch_match.group(1))
    
    train_loss_match = re.search(r'Train .*?loss: ([\d\.]+)', output_line)
    if train_loss_match:
        metrics['train_loss'] = float(train_loss_match.group(1))
    
    valid_loss_match = re.search(r'Validation .*?loss: ([\d\.]+)', output_line)
    if valid_loss_match:
        metrics['valid_loss'] = float(valid_loss_match.group(1))
    
    lr_match = re.search(r'lr: ([\d\.e\-]+)', output_line)
    if lr_match:
        metrics['learning_rate'] = float(lr_match.group(1))
    
    acc_match = re.search(r'accuracy: ([\d\.]+)', output_line)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))
    
    ppl_match = re.search(r'ppl: ([\d\.]+)', output_line)
    if ppl_match:
        metrics['perplexity'] = float(ppl_match.group(1))
    
    checkpoint_match = re.search(r'Saving checkpoint (.*?) to', output_line)
    if checkpoint_match:
        metrics['checkpoint'] = checkpoint_match.group(1)
    
    return metrics

def find_latest_checkpoint(models_dir=None):
    """Find the latest checkpoint in the models directory."""
    if models_dir is None:
        models_dir = os.path.join(project_root, 'models_rnn_big')
    
    checkpoint_pattern = os.path.join(models_dir, '*.pt')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    checkpoint_steps = []
    for checkpoint in checkpoints:
        step_match = re.search(r'_step_(\d+)\.pt$', checkpoint)
        if step_match:
            step = int(step_match.group(1))
            checkpoint_steps.append((step, checkpoint))
    
    if not checkpoint_steps:
        return None
    
    checkpoint_steps.sort(reverse=True)
    return checkpoint_steps[0][1]

def log_hyperparameters(config_path, logger):
    """Extract and log hyperparameters from the config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        hyperparams = {
            'model_architecture': {
                'encoder_type': config.get('encoder_type', 'N/A'),
                'decoder_type': config.get('decoder_type', 'N/A'),
                'enc_layers': config.get('enc_layers', 'N/A'),
                'dec_layers': config.get('dec_layers', 'N/A'),
                'heads': config.get('heads', 'N/A'),
                'hidden_size': config.get('hidden_size', 'N/A'),
                'word_vec_size': config.get('word_vec_size', 'N/A'),
                'transformer_ff': config.get('transformer_ff', 'N/A'),
            },
            'training_params': {
                'batch_size': config.get('batch_size', 'N/A'),
                'batch_type': config.get('batch_type', 'N/A'),
                'optim': config.get('optim', 'N/A'),
                'learning_rate': config.get('learning_rate', 'N/A'),
                'max_grad_norm': config.get('max_grad_norm', 'N/A'),
                'dropout': config.get('dropout', 'N/A'),
                'label_smoothing': config.get('label_smoothing', 'N/A'),
                'train_steps': config.get('train_steps', 'N/A'),
                'valid_steps': config.get('valid_steps', 'N/A'),
                'warmup_steps': config.get('warmup_steps', 'N/A'),
                'seed': config.get('seed', 'N/A'),
            }
        }
        
        logger.log_info("=== Hyperparameters ===")
        logger.log_info(f"Model Architecture: {json.dumps(hyperparams['model_architecture'], indent=2)}")
        logger.log_info(f"Training Parameters: {json.dumps(hyperparams['training_params'], indent=2)}")
        
        hyperparams_path = os.path.join(logger.log_dir, 'hyperparameters.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        
        logger.log_info(f"Hyperparameters saved to {hyperparams_path}")
        return hyperparams
    except Exception as e:
        logger.log_error(f"Error extracting hyperparameters: {e}")
        return {}

def monitor_resources(logger, gpu_id=0, interval=60):
    """Monitor system resources and log them."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024 ** 3)
    memory_total_gb = memory.total / (1024 ** 3)
    memory_percent = memory.percent
    
    gpu_metrics = {}
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                gpu = gpus[gpu_id]
                gpu_metrics = {
                    'gpu_load': gpu.load * 100,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_temperature': gpu.temperature
                }
        except Exception as e:
            logger.log_warning(f"Error getting GPU metrics: {e}")
    
    resource_metrics = {
        'cpu_percent': cpu_percent,
        'memory_used_gb': memory_used_gb,
        'memory_total_gb': memory_total_gb,
        'memory_percent': memory_percent,
        'timestamp': time.time()
    }
    
    if gpu_metrics:
        resource_metrics.update(gpu_metrics)
    
    logger.log_metrics(resource_metrics, save=True)
    return resource_metrics

def main():
    args = parse_args()
    
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = NMTLogger(log_dir=log_dir, experiment_name=args.experiment_name)
    logger.log_info(f"Starting OpenNMT training experiment: {args.experiment_name}")
    
    config_path = args.config
    logger.log_info(f"Using config file: {config_path}")
    
    checkpoint_path = None
    if args.continue_from_last:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            logger.log_info(f"Continuing training from latest checkpoint: {checkpoint_path}")
        else:
            logger.log_warning("No checkpoint found to continue from. Starting fresh training.")
    elif args.train_from:
        checkpoint_path = args.train_from
        if os.path.exists(checkpoint_path):
            logger.log_info(f"Continuing training from specified checkpoint: {checkpoint_path}")
        else:
            logger.log_error(f"Specified checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    
    hyperparams = log_hyperparameters(config_path, logger)
    
    system_info = {
        'python_version': sys.version,
        'os': os.name,
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3)
    }
    
    if GPU_AVAILABLE:
        try:
            system_info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info['cuda_version'] = torch.version.cuda
                system_info['gpu_name'] = torch.cuda.get_device_name(args.gpu)
                system_info['gpu_count'] = torch.cuda.device_count()
        except Exception as e:
            logger.log_warning(f"Error getting CUDA information: {e}")
    
    logger.log_info(f"System Information: {json.dumps(system_info, indent=2)}")
    
    train_cmd = [
        'onmt_train',
        '-config', config_path,
        '--gpu_ranks', str(args.gpu),
        '--report_every', '100',
        '--log_file', os.path.join(log_dir, 'onmt_train.log'),
        '--log_level', args.log_level
    ]
    
    if checkpoint_path:
        train_cmd.extend(['--train_from', checkpoint_path])
    
    logger.log_info("Running training command: " + ' '.join(train_cmd))
    
    if args.monitor_resources:
        logger.log_info(f"Resource monitoring enabled with interval {args.monitor_interval} seconds")
        last_monitor_time = 0
    
    try:
        process = subprocess.Popen(
            train_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            
            metrics = parse_onmt_output(line)
            if metrics and 'step' in metrics:
                logger.log_metrics(metrics)
                
                if args.monitor_resources and time.time() - last_monitor_time > args.monitor_interval:
                    resource_metrics = monitor_resources(logger, args.gpu, args.monitor_interval)
                    logger.log_info(f"Resource usage - CPU: {resource_metrics['cpu_percent']:.1f}%, "
                                   f"Memory: {resource_metrics['memory_percent']:.1f}%"
                                   + (f", GPU: {resource_metrics.get('gpu_load', 0):.1f}%" if 'gpu_load' in resource_metrics else ""))
                    last_monitor_time = time.time()
        
        return_code = process.wait()
        if return_code != 0:
            logger.log_error(f"Training process exited with code {return_code}")
            sys.exit(return_code)
        
        logger.log_info("Training completed successfully!")
        
        plots_dir = os.path.join(log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        try:
            logger.plot_metrics(save_dir=plots_dir)
            logger.log_info(f"Training plots saved to {plots_dir}")
        except Exception as e:
            logger.log_warning(f"Error generating plots: {e}")
            logger.log_info("Training completed successfully, but plots could not be generated.")
        
    except subprocess.CalledProcessError as e:
        logger.log_error(f"Error running training command: {e}")
        sys.exit(1)
    except Exception as e:
        logger.log_error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 