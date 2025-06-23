import argparse
import subprocess
import sys
import torch
import os
from accelerate import Accelerator

def run_baseline(args):
    cmd = [
        sys.executable, 'train_unet.py',
        '--data_root', args.data_root,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--experiment_dir', args.experiment_dir,
        '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
        '--mixed_precision', args.mixed_precision,
        '--modalities', args.modalities
    ]
    
    if args.seed is not None:
        cmd.extend(['--seed', str(args.seed)])
    
    subprocess.run(cmd)

def run_finetune(args):
    cmd = [
        sys.executable, 'finetune_ct.py',
        '--pretrained_model', args.pretrained_model,
        '--data_root', args.data_root,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--experiment_dir', args.experiment_dir,
        '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
        '--mixed_precision', args.mixed_precision,
        '--modalities', args.modalities
    ]
    
    if args.seed is not None:
        cmd.extend(['--seed', str(args.seed)])
    
    if args.freeze_encoder_epoch is not None:
        cmd.extend(['--freeze_encoder_epoch', str(args.freeze_encoder_epoch)])
    
    subprocess.run(cmd)

def run_finetune(args):
    cmd = [
        sys.executable, 'finetune_ct.py',
        '--pretrained_model', args.pretrained_model,
        '--data_root', args.data_root,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--experiment_dir', args.experiment_dir,
        '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
        '--mixed_precision', args.mixed_precision,
        '--modalities', args.modalities
    ]
    
    if args.seed is not None:
        cmd.extend(['--seed', str(args.seed)])
    
     if args.freeze_encoder:
        cmd.append('--freeze_encoder')
        
    subprocess.run(cmd)



def run_eval(args):
    if args.model_path is None:
        raise ValueError("--model_path is required for evaluation experiments")
    
    cmd = [
        sys.executable, 'test_model.py',
        '--model_path', args.model_path,
        '--data_root', args.data_root,
        '--experiment_dir', args.experiment_dir,
        '--model_name', args.model_name,
        '--modalities', args.modalities
    ]
    
    subprocess.run(cmd)

def run_eval(args):
    if args.model_path is None:
        raise ValueError("--model_path is required for evaluation experiments")
    
    cmd = [
        sys.executable, 'test_model.py',
        '--model_path', args.model_path,
        '--data_root', args.data_root,
        '--experiment_dir', args.experiment_dir,
        '--model_name', args.model_name,
        '--modalities', args.modalities
    ]
    
    subprocess.run(cmd)

def main():
    # Initialize accelerator
    accelerator = Accelerator()

    # Print GPU information only on main process
    if accelerator.is_main_process:
        print("\n=== GPU Information ===")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        print(f"GPU device name: {torch.cuda.get_device_name()}")
        print(f"Process ID: {os.getpid()}")
        print("=====================\n")

    parser = argparse.ArgumentParser(description='Orchestrate multimodal segmentation experiments')
    parser.add_argument('--experiment', type=str, default='train', 
                       choices=['train', 'finetune', 'eval', 'transfer', 'dann', 'distill', 'cyclegan'], 
                       help='Experiment type')
    parser.add_argument('--data_root', type=str, default='datasets/resampled', 
                       help='Root directory of dataset splits')
    parser.add_argument('--batch_size', type=int, default=2, 
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--experiment_dir', type=str, default='experiments', 
                       help='Directory to save experiments')
    parser.add_argument('--modalities', type=str, default='all',
                       help='Comma-separated list of modalities to include (e.g., "ct", "mri", "ct,mri", "all")')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer')
    
    # Fine-tuning specific arguments
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Path to pre-trained model checkpoint (required for fine-tuning)')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder layers during fine-tuning to prevent overfitting')
    parser.add_argument('--freeze_encoder_epoch', type=int, default=None,
                       help='Epoch to freeze the encoder (for training experiments)')
    
    # Evaluation specific arguments
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint (required for evaluation)')
    parser.add_argument('--model_name', type=str, default='unet',
                       help='Name of the model for result folder (required for evaluation)')
    
    # Accelerate specific arguments
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of steps to accumulate gradients')
    parser.add_argument('--mixed_precision', type=str, default='no',
                       choices=['no', 'fp16', 'bf16'],
                       help='Mixed precision training type')
    
    args = parser.parse_args()

    if args.experiment == 'train':
        run_baseline(args)
    elif args.experiment == 'finetune':
        if args.pretrained_model is None:
            raise ValueError("--pretrained_model is required for fine-tuning experiments")
        run_finetune(args)
    elif args.experiment == 'eval':
        run_eval(args)
    elif args.experiment == 'transfer':
        # TODO: Implement transfer learning script call
        print("Transfer learning not implemented yet.")
    elif args.experiment == 'dann':
        # TODO: Implement DANN script call
        print("DANN not implemented yet.")
    elif args.experiment == 'distill':
        # TODO: Implement knowledge distillation script call
        print("Knowledge distillation not implemented yet.")
    elif args.experiment == 'cyclegan':
        # TODO: Implement CycleGAN script call
        print("CycleGAN not implemented yet.")
    else:
        raise NotImplementedError(f"Experiment type {args.experiment} not implemented yet")

if __name__ == "__main__":
    main() 