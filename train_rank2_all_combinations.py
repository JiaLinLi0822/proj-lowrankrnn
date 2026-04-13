"""
Train Rank 2 RNN models on all task combinations and training regimes.

This script trains models for:
- Task combinations: flipflop+cycle, flipflop+line, line+cycle (3 combinations)
- Training regimes: gated (standard), orthogonal, parallel (3 regimes)
- Total: 9 combinations
- Instances: 3 instances per combination

Supports parallel training to speed up the process.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
from datetime import datetime, timedelta

from data.custom_data_generator import (
    # Gated (standard) generators
    FlipFlopCycleGenerator,
    FlipFlopLineGenerator,
    LimitCycleLineGenerator,
    # Orthogonal generators
    OrthogonalFlipFlopCycleGenerator,
    OrthogonalFlipFlopLineGenerator,
    OrthogonalCycleLineGenerator,
    # Parallel generators
    ParallelFlipFlopCycleGenerator,
    ParallelFlipFlopLineGenerator,
    ParallelCycleLineGenerator,
)
from model.model_wrapper import ModelWrapper, OptimizationParameters
from model.pt_models import Rank2Architechture
import tools.pytorchtools as pytorchtools


# Configuration
UNITS = 100
N_BITS = 2
N_INSTANCES = 3
INSTANCE_START = 0  # Starting instance number

# Training parameters
OPTIMIZATION_PARAMS = OptimizationParameters(
    batch_size=32,
    epochs=10,
    minimal_loss=1e-4,
    initial_lr=1e-4
)

# Device configuration
DEVICE = 'cpu'  # Change to 'mps' for Apple Silicon or 'cuda' for GPU

# Parallel training configuration
USE_PARALLEL = True  # Set to False for sequential training
MAX_WORKERS = None  # None = use all available CPUs, or specify a number


# Define all combinations
TASK_COMBINATIONS = {
    'flipflop_cycle': {
        'gated': FlipFlopCycleGenerator,
        'orthogonal': OrthogonalFlipFlopCycleGenerator,
        'parallel': ParallelFlipFlopCycleGenerator,
    },
    'flipflop_line': {
        'gated': FlipFlopLineGenerator,
        'orthogonal': OrthogonalFlipFlopLineGenerator,
        'parallel': ParallelFlipFlopLineGenerator,
    },
    'line_cycle': {
        'gated': LimitCycleLineGenerator,
        'orthogonal': OrthogonalCycleLineGenerator,
        'parallel': ParallelCycleLineGenerator,
    },
}

TRAINING_REGIMES = ['gated', 'orthogonal', 'parallel']


def set_device(device_type='cpu'):
    """Set the training device."""
    import torch
    
    if device_type == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            pytorchtools.device = torch.device("mps")
            print("Device set to: MPS (Apple Silicon GPU)")
        else:
            print("Warning: MPS not available, falling back to CPU")
            pytorchtools.device = torch.device("cpu")
    elif device_type == 'cuda':
        if torch.cuda.is_available():
            pytorchtools.device = torch.device("cuda")
            print("Device set to: CUDA")
        else:
            print("Warning: CUDA not available, falling back to CPU")
            pytorchtools.device = torch.device("cpu")
    else:
        pytorchtools.device = torch.device("cpu")
        print("Device set to: CPU")


def train_combination_worker(args):
    """
    Worker function for parallel training. This function is called in a separate process.
    
    Args:
        args: Tuple of (task_combo_name, training_regime, instance_start, n_instances, device)
    
    Returns:
        Tuple of (task_combo_name, training_regime, success, message)
    """
    task_combo_name, training_regime, instance_start, n_instances, device = args
    
    try:
        import torch
        import tools.pytorchtools as pytorchtools
        
        # Re-import necessary modules in worker process
        from data.custom_data_generator import (
            FlipFlopCycleGenerator,
            FlipFlopLineGenerator,
            LimitCycleLineGenerator,
            OrthogonalFlipFlopCycleGenerator,
            OrthogonalFlipFlopLineGenerator,
            OrthogonalCycleLineGenerator,
            ParallelFlipFlopCycleGenerator,
            ParallelFlipFlopLineGenerator,
            ParallelCycleLineGenerator,
        )
        from model.model_wrapper import ModelWrapper, OptimizationParameters
        from model.pt_models import Rank2Architechture
        
        # Recreate TASK_COMBINATIONS dict in worker process
        task_combinations = {
            'flipflop_cycle': {
                'gated': FlipFlopCycleGenerator,
                'orthogonal': OrthogonalFlipFlopCycleGenerator,
                'parallel': ParallelFlipFlopCycleGenerator,
            },
            'flipflop_line': {
                'gated': FlipFlopLineGenerator,
                'orthogonal': OrthogonalFlipFlopLineGenerator,
                'parallel': ParallelFlipFlopLineGenerator,
            },
            'line_cycle': {
                'gated': LimitCycleLineGenerator,
                'orthogonal': OrthogonalCycleLineGenerator,
                'parallel': ParallelCycleLineGenerator,
            },
        }
        
        # Set device for this process
        set_device(device)
        
        # Get the appropriate generator class
        generator_class = task_combinations[task_combo_name][training_regime]
        
        # Create data generator
        data_generator = generator_class(n_bits=2)  # N_BITS
        
        # Create model wrapper
        optimization_params = OptimizationParameters(
            batch_size=32,
            epochs=10,
            minimal_loss=1e-4,
            initial_lr=1e-4
        )
        
        wrapper = ModelWrapper(
            architecture_func=Rank2Architechture,
            units=100,  # UNITS
            train_data=data_generator,
            optimization_params=optimization_params,
            instance_range=range(instance_start, instance_start + n_instances),
            recurrent_bias=False,
            readout_bias=True
        )
        
        # Train model
        wrapper.train_model()
        
        # Return success
        model_paths = [f"models/{wrapper.name}/i{inst}/weights.pt" 
                      for inst in range(instance_start, instance_start + n_instances)]
        return (task_combo_name, training_regime, True, f"Saved to: {model_paths[0]}...")
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        return (task_combo_name, training_regime, False, error_msg)


def train_combination(task_combo_name, training_regime, instance_start=0, n_instances=3, device='cpu'):
    """
    Train a specific task combination and training regime.
    
    Args:
        task_combo_name: Name of task combination ('flipflop_cycle', 'flipflop_line', 'line_cycle')
        training_regime: Training regime ('gated', 'orthogonal', 'parallel')
        instance_start: Starting instance number
        n_instances: Number of instances to train
        device: Device to use for training
    """
    print("\n" + "="*80)
    print(f"Training: {task_combo_name} - {training_regime}")
    print("="*80)
    
    # Set device
    set_device(device)
    
    # Get the appropriate generator class
    generator_class = TASK_COMBINATIONS[task_combo_name][training_regime]
    
    # Create data generator
    data_generator = generator_class(n_bits=N_BITS)
    
    # Create model wrapper
    wrapper = ModelWrapper(
        architecture_func=Rank2Architechture,
        units=UNITS,
        train_data=data_generator,
        optimization_params=OPTIMIZATION_PARAMS,
        instance_range=range(instance_start, instance_start + n_instances),
        recurrent_bias=False,
        readout_bias=True
    )
    
    # Train model
    print(f"\nTraining {n_instances} instances (i{instance_start} to i{instance_start + n_instances - 1})...")
    wrapper.train_model()
    
    # Print model save locations
    print(f"\nModels saved to:")
    for inst in range(instance_start, instance_start + n_instances):
        model_path = f"models/{wrapper.name}/i{inst}"
        print(f"  Instance {inst}: {model_path}/weights.pt")
    
    return wrapper


def main():
    """Main function to train all combinations."""
    import torch
    
    # Configuration - modify these values directly in the script
    device_to_use = DEVICE
    use_parallel = USE_PARALLEL
    max_workers = MAX_WORKERS
    
    print("\n" + "="*80)
    print("Rank 2 RNN Training: All Task Combinations")
    print("="*80)
    print(f"Task combinations: {list(TASK_COMBINATIONS.keys())}")
    print(f"Training regimes: {TRAINING_REGIMES}")
    print(f"Total combinations: {len(TASK_COMBINATIONS) * len(TRAINING_REGIMES)}")
    print(f"Instances per combination: {N_INSTANCES}")
    print(f"Total models to train: {len(TASK_COMBINATIONS) * len(TRAINING_REGIMES) * N_INSTANCES}")
    print(f"Device: {device_to_use}")
    print(f"Parallel training: {use_parallel}")
    if use_parallel:
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        print(f"Max parallel workers: {max_workers}")
    print("="*80)
    
    # Prepare all training tasks
    all_tasks = []
    for task_combo_name in TASK_COMBINATIONS.keys():
        for training_regime in TRAINING_REGIMES:
            all_tasks.append((task_combo_name, training_regime, INSTANCE_START, N_INSTANCES, device_to_use))
    
    total_combinations = len(all_tasks)
    
    # Time tracking
    start_time = time.time()
    task_times = []  # Store time taken for each completed task
    
    def format_time(seconds):
        """Format seconds into human-readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def estimate_remaining(completed, total, task_times):
        """Estimate remaining time based on completed tasks."""
        if completed == 0 or len(task_times) == 0:
            return None, None
        
        # Calculate average time per task
        avg_time = sum(task_times) / len(task_times)
        remaining_tasks = total - completed
        
        # For parallel training, account for parallelization
        if use_parallel:
            # Effective tasks remaining considering parallel workers
            effective_remaining = remaining_tasks / max_workers
            estimated_seconds = avg_time * effective_remaining
        else:
            estimated_seconds = avg_time * remaining_tasks
        
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        return estimated_seconds, estimated_completion
    
    if use_parallel:
        # Parallel training
        print(f"\nStarting parallel training with {max_workers} workers...")
        print("="*80)
        
        completed = 0
        failed = []
        task_start_times = {}  # Track when each task started
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(train_combination_worker, task): task 
                for task in all_tasks
            }
            
            # Record start time for all tasks
            for future, task in future_to_task.items():
                task_start_times[future] = time.time()
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_combo_name, training_regime = task[0], task[1]
                completed += 1
                
                # Calculate time taken for this task
                task_time = time.time() - task_start_times[future]
                task_times.append(task_time)
                
                try:
                    result = future.result()
                    combo_name, regime, success, message = result
                    
                    # Estimate remaining time
                    est_seconds, est_completion = estimate_remaining(completed, total_combinations, task_times)
                    
                    if success:
                        status_msg = f"[{completed}/{total_combinations}] ✓ {combo_name} - {regime}"
                        if est_seconds is not None:
                            status_msg += f" | Time: {format_time(task_time)} | ETA: {est_completion.strftime('%H:%M:%S')} ({format_time(est_seconds)} remaining)"
                        else:
                            status_msg += f" | Time: {format_time(task_time)}"
                        print(status_msg)
                    else:
                        print(f"[{completed}/{total_combinations}] ✗ {combo_name} - {regime}: {message}")
                        failed.append((combo_name, regime))
                except Exception as e:
                    print(f"[{completed}/{total_combinations}] ✗ {task_combo_name} - {training_regime}: Exception - {e}")
                    failed.append((task_combo_name, training_regime))
        
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("Parallel training completed!")
        print("="*80)
        print(f"Total time: {format_time(total_time)}")
        if task_times:
            print(f"Average time per combination: {format_time(sum(task_times) / len(task_times))}")
        if failed:
            print(f"\nFailed combinations ({len(failed)}):")
            for combo, regime in failed:
                print(f"  - {combo} - {regime}")
        else:
            print("\nAll combinations trained successfully!")
    
    else:
        # Sequential training
        current_combination = 0
        
        for task_combo_name, training_regime, instance_start, n_instances, device in all_tasks:
            current_combination += 1
            task_start = time.time()
            print(f"\n[{current_combination}/{total_combinations}] ", end="")
            
            try:
                wrapper = train_combination(
                    task_combo_name=task_combo_name,
                    training_regime=training_regime,
                    instance_start=instance_start,
                    n_instances=n_instances,
                    device=device
                )
                task_time = time.time() - task_start
                task_times.append(task_time)
                
                # Estimate remaining time
                est_seconds, est_completion = estimate_remaining(current_combination, total_combinations, task_times)
                
                status_msg = f"✓ Successfully trained {task_combo_name} - {training_regime} | Time: {format_time(task_time)}"
                if est_seconds is not None:
                    status_msg += f" | ETA: {est_completion.strftime('%H:%M:%S')} ({format_time(est_seconds)} remaining)"
                print(status_msg)
            except Exception as e:
                task_time = time.time() - task_start
                task_times.append(task_time)
                print(f"✗ Error training {task_combo_name} - {training_regime}: {e}")
                import traceback
                traceback.print_exc()
                print("\nContinuing with next combination...")
        
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)
        print(f"Total time: {format_time(total_time)}")
        if task_times:
            print(f"Average time per combination: {format_time(sum(task_times) / len(task_times))}")
    
    print(f"\nAll models saved in: models/")
    print("\nModel naming convention:")
    print("  models/{generator_name}/i{instance}/weights.pt")
    print("\nExample paths:")
    for task_combo_name in list(TASK_COMBINATIONS.keys())[:1]:
        for training_regime in TRAINING_REGIMES[:1]:
            generator_class = TASK_COMBINATIONS[task_combo_name][training_regime]
            gen = generator_class(n_bits=N_BITS)
            print(f"  models/{gen.name}/i0/weights.pt")


if __name__ == '__main__':
    main()

