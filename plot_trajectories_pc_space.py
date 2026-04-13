"""
Script to load a rank 2 gated model trained with flipflop and line tasks,
run 100 example trials, and plot trajectories in PC space.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from model.pt_models import Rank2Architechture
from model.model_wrapper import load_weights
from data.custom_data_generator import (
    FlipFlopLineGenerator,
    FlipFlopCycleGenerator,
    LimitCycleLineGenerator,
    OrthogonalFlipFlopLineGenerator,
    OrthogonalFlipFlopCycleGenerator,
    OrthogonalCycleLineGenerator,
    ParallelFlipFlopLineGenerator,
    ParallelFlipFlopCycleGenerator,
    ParallelCycleLineGenerator
)
import tools.pytorchtools as pytorchtools

plt.rcParams.update({
        'font.family': 'Arial',
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7
    })

def find_model_path(task_combination, training_mode="standard", rank=2, instance=100):
    """
    Find model path matching the task combination and training mode.
    
    Args:
        task_combination: List of task names, e.g., ['flipflop', 'line'] or ['flipflop', 'cycle']
        training_mode: 'standard', 'orthogonal', or 'parallel'
        rank: Rank of the model (2 for rank 2)
        instance: Instance number
        
    Returns:
        Path to model weights file or None
    """
    models_dir = Path("models")
    if not models_dir.exists():
        return None
    
    # Build search pattern based on task combination and training mode
    tasks_sorted = sorted(task_combination)  # Sort for consistent matching
    
    # Create pattern from task names
    if len(tasks_sorted) == 2:
        if 'flipflop' in tasks_sorted and 'line' in tasks_sorted:
            base_pattern = "flipflopline"
        elif 'flipflop' in tasks_sorted and 'cycle' in tasks_sorted:
            base_pattern = "flipflopcycle"
        elif 'cycle' in tasks_sorted and 'line' in tasks_sorted:
            base_pattern = "cycleline"  # Note: might be "cycle_line" or "limitcycleline"
        else:
            base_pattern = "".join(tasks_sorted)
    else:
        base_pattern = "".join(tasks_sorted)
    
    # Add training mode prefix
    if training_mode == "orthogonal":
        pattern = f"{base_pattern}orthogonal" if "orthogonal" not in base_pattern else base_pattern
    elif training_mode == "parallel":
        pattern = f"{base_pattern}parallel" if "parallel" not in base_pattern else base_pattern
    else:  # standard
        pattern = base_pattern
    
    # Search for models matching the pattern
    for model_folder in models_dir.iterdir():
        if model_folder.is_dir():
            model_name = model_folder.name.lower()
            # Check if all task names are in the model name
            all_tasks_match = all(task in model_name for task in tasks_sorted)
            if all_tasks_match and f"rank{rank}" in model_name:
                # Additional check for training mode
                if training_mode == "orthogonal" and "orthogonal" not in model_name:
                    continue
                elif training_mode == "parallel" and "parallel" not in model_name:
                    continue
                elif training_mode == "standard" and ("orthogonal" in model_name or "parallel" in model_name):
                    continue
                
                instance_dir = model_folder / f"i{instance}"
                weights_file = instance_dir / "weights.pt"
                if weights_file.exists():
                    return str(weights_file)
    
    return None


def get_data_generator(task_combination, training_mode="standard", n_bits=2):
    """
    Get the appropriate data generator for the task combination.
    
    Args:
        task_combination: List of task names, e.g., ['flipflop', 'line'] or ['flipflop', 'cycle']
        training_mode: 'standard', 'orthogonal', or 'parallel'
        n_bits: Number of bits for the tasks
        
    Returns:
        Data generator instance
    """
    tasks_sorted = sorted(task_combination)
    
    # Determine generator class based on task combination and training mode
    if len(tasks_sorted) == 2:
        if 'flipflop' in tasks_sorted and 'line' in tasks_sorted:
            if training_mode == "orthogonal":
                return OrthogonalFlipFlopLineGenerator(n_bits=n_bits)
            elif training_mode == "parallel":
                return ParallelFlipFlopLineGenerator(n_bits=n_bits)
            else:
                return FlipFlopLineGenerator(n_bits=n_bits)
        
        elif 'flipflop' in tasks_sorted and 'cycle' in tasks_sorted:
            if training_mode == "orthogonal":
                return OrthogonalFlipFlopCycleGenerator(n_bits=n_bits)
            elif training_mode == "parallel":
                return ParallelFlipFlopCycleGenerator(n_bits=n_bits)
            else:
                return FlipFlopCycleGenerator(n_bits=n_bits)
        
        elif 'cycle' in tasks_sorted and 'line' in tasks_sorted:
            if training_mode == "orthogonal":
                return OrthogonalCycleLineGenerator(n_bits=n_bits)
            elif training_mode == "parallel":
                return ParallelCycleLineGenerator(n_bits=n_bits)
            else:
                return LimitCycleLineGenerator(n_bits=n_bits)
    
    raise ValueError(f"Unsupported task combination: {task_combination}")


def load_model_and_data(model_path, units=100, n_bits=2, steps=200, task_combination=None, training_mode="standard"):
    """
    Load the model and create data generator.
    
    Args:
        model_path: Path to model weights file
        units: Number of hidden units
        n_bits: Number of bits for the tasks
        steps: Number of time steps
        task_combination: List of task names, e.g., ['flipflop', 'line']
        training_mode: 'standard', 'orthogonal', or 'parallel'
        
    Returns:
        model: Loaded model architecture
        data_generator: Data generator for the specified tasks
        weights: Model weights
    """
    # Load weights
    weights = load_weights(Path(model_path).parent, "weights", map_location='cpu')
    
    # Create data generator based on task combination and training mode
    if task_combination is None:
        task_combination = ['flipflop', 'line']  # Default
    
    data_generator = get_data_generator(task_combination, training_mode, n_bits)
    
    # Set steps after initialization
    data_generator._steps = steps
    
    # Create model architecture
    model = Rank2Architechture(
        units=units,
        inputs=data_generator.n_inputs,
        outputs=data_generator.n_outputs,
        recurrent_bias=False,
        readout_bias=True
    )
    
    # Set model directory (needed for some internal operations)
    model.set_model_dir(str(Path(model_path).parent))
    model.load_weights(weights)
    
    return model, data_generator, weights


def identify_task_type(x, data_generator, task_combination, training_mode="standard"):
    """
    Identify task type from input.
    
    Args:
        x: Input array of shape (time_steps, n_inputs)
        data_generator: Data generator with vmin and vmax
        task_combination: List of task names, e.g., ['flipflop', 'line']
        training_mode: 'standard', 'orthogonal', or 'parallel'
        
    Returns:
        task_label: Task name (e.g., 'flipflop', 'line', 'cycle')
        task_num: Task number (0, 1, etc.)
    """
    tasks_sorted = sorted(task_combination)
    n_tasks = len(tasks_sorted)
    
    # For 2-task combinations, identify which task is active
    if n_tasks == 2:
        # Check which channels are active
        channel_activities = [np.sum(np.abs(x[:, i]) > 1e-6) for i in range(n_tasks)]
        
        if training_mode == "parallel":
            # In parallel mode, both channels may be active
            # Use heuristics to identify the dominant task
            vmin, vmax = data_generator.vmin, data_generator.vmax
            
            # Check for binary values (flipflop) vs continuous/wave (cycle/line)
            channel_binary_counts = []
            for i in range(n_tasks):
                binary_count = np.sum(np.isclose(x[:, i], vmin) | np.isclose(x[:, i], vmax))
                channel_binary_counts.append(binary_count)
            
            # Determine task based on patterns
            if 'flipflop' in tasks_sorted:
                flipflop_idx = tasks_sorted.index('flipflop')
                if channel_binary_counts[flipflop_idx] > max([c for i, c in enumerate(channel_binary_counts) if i != flipflop_idx]):
                    return 'flipflop', flipflop_idx
            
            # Check for wave pattern (cycle) - typically more oscillatory
            if 'cycle' in tasks_sorted:
                cycle_idx = tasks_sorted.index('cycle')
                # Cycle tasks typically have wave outputs, check output variance
                if channel_activities[cycle_idx] > 0:
                    return 'cycle', cycle_idx
            
            # Default to line if present
            if 'line' in tasks_sorted:
                line_idx = tasks_sorted.index('line')
                if channel_activities[line_idx] > 0:
                    return 'line', line_idx
            
            # Fallback: most active channel
            dominant_idx = np.argmax(channel_activities)
            return tasks_sorted[dominant_idx], dominant_idx
        else:
            # For standard and orthogonal modes, typically only one channel is active
            active_channels = [i for i in range(n_tasks) if channel_activities[i] > 0]
            
            if len(active_channels) == 1:
                task_idx = active_channels[0]
                return tasks_sorted[task_idx], task_idx
            elif len(active_channels) > 1:
                # Multiple channels active, use the most active one
                dominant_idx = active_channels[np.argmax([channel_activities[i] for i in active_channels])]
                return tasks_sorted[dominant_idx], dominant_idx
            else:
                # No clear activity, default to first task
                return tasks_sorted[0], 0
    
    # Default fallback
    return tasks_sorted[0], 0
    # For parallel mode, both channels may be active simultaneously
    # We need to check which task is more dominant or use a different strategy
    
    if training_mode == "parallel":
        # In parallel mode, both tasks can be active
        # Check which channel has more distinct binary values (flipflop) vs continuous (line)
        channel0_activity = np.sum(np.abs(x[:, 0]) > 1e-6)
        channel1_activity = np.sum(np.abs(x[:, 1]) > 1e-6)
        
        # Check if values are binary (vmin or vmax) for flipflop
        vmin, vmax = data_generator.vmin, data_generator.vmax
        channel0_binary = np.sum(np.isclose(x[:, 0], vmin) | np.isclose(x[:, 0], vmax))
        channel1_binary = np.sum(np.isclose(x[:, 1], vmin) | np.isclose(x[:, 1], vmax))
        
        # If channel 0 has more binary values, it's likely flipflop
        # If channel 1 has more continuous values, it's likely line
        if channel0_activity > 0 and channel0_binary > channel1_binary:
            return 'flipflop', 0
        elif channel1_activity > 0:
            return 'line', 1
        else:
            # Default based on activity
            return 'flipflop', 0 if channel0_activity >= channel1_activity else 'line', 1
    else:
        # For standard and orthogonal modes, only one channel is typically active
        channel0_active = np.any(np.abs(x[:, 0]) > 1e-6)
        channel1_active = np.any(np.abs(x[:, 1]) > 1e-6)
        
        if channel0_active and not channel1_active:
            return 'flipflop', 0
        elif channel1_active and not channel0_active:
            return 'line', 1
        else:
            # If both active, check which one has more activity
            if np.sum(np.abs(x[:, 0])) > np.sum(np.abs(x[:, 1])):
                return 'flipflop', 0
            else:
                return 'line', 1


def run_trials(model, data_generator, task_combination, n_trials_per_task=100, training_mode="standard"):
    """
    Run trials and collect state trajectories with task labels.
    Ensures equal number of trials for each task type.
    
    Args:
        model: Model architecture
        data_generator: Data generator
        task_combination: List of task names, e.g., ['flipflop', 'line']
        n_trials_per_task: Number of trials per task type (default: 100)
        training_mode: 'standard', 'orthogonal', or 'parallel'
        
    Returns:
        states: Array of shape (n_trials, time_steps, hidden_units)
        inputs: Array of shape (n_trials, time_steps, n_inputs)
        task_labels: List of task labels
    """
    all_states = []
    all_inputs = []
    task_labels = []
    
    n_task_types = len(task_combination)
    total_trials = n_trials_per_task * n_task_types
    
    # Initialize counters for each task type
    task_counts = {task: 0 for task in task_combination}
    
    print(f"Running {total_trials} trials ({n_trials_per_task} per task type: {', '.join(task_combination)})...")
    
    # Generate trials until we have enough of each type
    max_attempts = total_trials * 3  # Safety limit
    attempts = 0
    
    while any(count < n_trials_per_task for count in task_counts.values()) and attempts < max_attempts:
        attempts += 1
        
        # Generate a trial
        x, y = data_generator.generate_training_trial()
        x = x[np.newaxis, :, :]  # Add batch dimension
        
        # Identify task type
        task_label, task_num = identify_task_type(x[0], data_generator, task_combination, training_mode=training_mode)
        
        # Only add if we need more of this task type
        if task_label in task_counts and task_counts[task_label] < n_trials_per_task:
            # Run model forward
            predictions = model.predict(x)
            states = predictions['state']  # Shape: (1, time_steps, hidden_units)
            
            all_states.append(states[0])  # Remove batch dimension
            all_inputs.append(x[0])
            task_labels.append(task_label)
            task_counts[task_label] += 1
        
        # Progress update
        if sum(task_counts.values()) % 20 == 0:
            progress_str = ", ".join([f"{task}={task_counts[task]}/{n_trials_per_task}" 
                                     for task in task_combination])
            print(f"  Progress: {progress_str}")
    
    if attempts >= max_attempts:
        print(f"Warning: Reached maximum attempts.")
        for task in task_combination:
            print(f"  {task}: {task_counts[task]}")
    
    states_array = np.array(all_states)  # (n_trials, time_steps, hidden_units)
    inputs_array = np.array(all_inputs)  # (n_trials, time_steps, n_inputs)
    
    final_str = ", ".join([f"{task}={task_counts[task]}" for task in task_combination])
    print(f"  Final: {final_str}, Total={len(task_labels)}")
    
    return states_array, inputs_array, task_labels


def extract_low_rank_basis(model, weights):
    """
    Extract U and V matrices (low-rank basis) from rank-2 model.
    
    Args:
        model: Model architecture
        weights: Model weights dictionary
        
    Returns:
        V: Matrix of shape (hidden_size, 2) - corresponds to 'm' in the code
        U: Matrix of shape (hidden_size, 2) - corresponds to 'n' in the code
    """
    # Try to get from weights dictionary first (most reliable)
    # Check various possible key names
    possible_v_keys = ['rnn.m', 'm', 'rnn.rnn.m']
    possible_u_keys = ['rnn.n', 'n', 'rnn.rnn.n']
    
    V = None
    U = None
    
    for key in possible_v_keys:
        if key in weights:
            V_tensor = weights[key]
            V = V_tensor.cpu().numpy() if torch.is_tensor(V_tensor) else V_tensor
            break
    
    for key in possible_u_keys:
        if key in weights:
            U_tensor = weights[key]
            U = U_tensor.cpu().numpy() if torch.is_tensor(U_tensor) else U_tensor
            break
    
    # If not found in weights, try to get from model instance
    if V is None or U is None:
        try:
            model_instance = model.get_model()
            if hasattr(model_instance, 'rnn'):
                rnn_module = model_instance.rnn
                if hasattr(rnn_module, 'm') and hasattr(rnn_module, 'n'):
                    if V is None:
                        V = rnn_module.m.detach().cpu().numpy()
                    if U is None:
                        U = rnn_module.n.detach().cpu().numpy()
        except:
            pass
    
    if V is None or U is None:
        print(f"Available weight keys: {list(weights.keys())[:10]}...")  # Print first 10 keys
        raise ValueError("Could not extract low-rank basis from model. Make sure it's a rank-2 model.")
    
    # W_rec = n @ m^T, so m corresponds to V, n corresponds to U
    # For projection: z_t = V^T @ h_t = m^T @ h_t
    return V, U


def project_to_low_rank_space(states, V):
    """
    Project states to 2D low-rank space: z_t = V^T @ h_t
    
    Args:
        states: Array of shape (n_trials, time_steps, hidden_units)
        V: Low-rank basis matrix of shape (hidden_size, 2)
        
    Returns:
        z: Projected states of shape (n_trials, time_steps, 2)
    """
    n_trials, time_steps, hidden_units = states.shape
    # Reshape for matrix multiplication: (n_trials * time_steps, hidden_units)
    states_flat = states.reshape(-1, hidden_units)
    # Project: z = h @ V (equivalent to V^T @ h)
    z_flat = states_flat @ V  # (n_trials * time_steps, 2)
    # Reshape back
    z = z_flat.reshape(n_trials, time_steps, 2)
    return z


def plot_low_rank_trajectories(states, task_labels, V, task_combination, save_path="trajectories_low_rank.png"):
    """
    Plot trajectories in low-rank 2D space (V^T @ h_t) with task-based coloring.
    
    Args:
        states: Array of shape (n_trials, time_steps, hidden_units)
        task_labels: List of task labels
        V: Low-rank basis matrix of shape (hidden_size, 2)
        task_combination: List of task names, e.g., ['flipflop', 'line']
        save_path: Path to save the plot
    """
    n_trials, time_steps, hidden_units = states.shape
    
    print(f"\nProjecting to low-rank space using V^T...")
    print(f"  V shape: {V.shape}")
    
    # Project states to 2D low-rank space
    trajectories_2d = project_to_low_rank_space(states, V)  # (n_trials, time_steps, 2)
    
    # Separate trajectories by task
    task_trajectories = {task: [] for task in task_combination}
    
    for i, task_label in enumerate(task_labels):
        if task_label in task_trajectories:
            task_trajectories[task_label].append(trajectories_2d[i])
    
    # Convert to arrays
    task_trajectories = {task: np.array(trajs) if trajs else None 
                        for task, trajs in task_trajectories.items()}
    
    # Compute average trajectories
    avg_trajectories = {}
    for task in task_combination:
        if task_trajectories[task] is not None:
            avg_trajectories[task] = task_trajectories[task].mean(axis=0)
        else:
            avg_trajectories[task] = None
    
    avg_all = trajectories_2d.mean(axis=0)
    
    # Print trial counts
    for task in task_combination:
        count = len(task_trajectories[task]) if task_trajectories[task] is not None else 0
        print(f"  {task.capitalize()} trials: {count}")
    
    # Define colors for different tasks
    task_colors = {
        'flipflop': ('blue', 'darkblue'),
        'line': ('orange', 'red'),
        'cycle': ('green', 'darkgreen')
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(3.35, 2.85))
    
    # Plot individual trajectories with task-based colors (thinner lines)
    for task in task_combination:
        if task_trajectories[task] is not None:
            color, _ = task_colors.get(task, ('gray', 'black'))
            # Plot first trajectory with label for legend (only show once per task)
            if len(task_trajectories[task]) > 0:
                ax.plot(task_trajectories[task][0][:, 0], task_trajectories[task][0][:, 1], 
                       alpha=0.2, color=color, linewidth=0.8, 
                       label=f'{task.capitalize()}', zorder=1)
            # Plot remaining trajectories without label
            for traj in task_trajectories[task][1:]:
                ax.plot(traj[:, 0], traj[:, 1], 
                       alpha=0.2, color=color, linewidth=0.8, zorder=1)
    
    # # Plot average trajectories
    # for task in task_combination:
    #     if avg_trajectories[task] is not None:
    #         _, avg_color = task_colors.get(task, ('gray', 'black'))
    #         ax.plot(avg_trajectories[task][:, 0], avg_trajectories[task][:, 1], 
    #                color=avg_color, linewidth=3, label=f'{task.capitalize()} (avg)', zorder=4)
    #         ax.scatter(avg_trajectories[task][0, 0], avg_trajectories[task][0, 1], 
    #                   color=avg_color, s=100, marker='o', zorder=5)
    #         ax.scatter(avg_trajectories[task][-1, 0], avg_trajectories[task][-1, 1], 
    #                   color=avg_color, s=100, marker='s', zorder=5)
    
    # # Plot overall average
    # ax.plot(avg_all[:, 0], avg_all[:, 1], 
    #        color='gray', linewidth=2, linestyle='--', alpha=0.7, label='Overall avg', zorder=3)
    
    # Create title with task names
    # task_str = ' + '.join([t.capitalize() for t in task_combination])
    # color_str = ', '.join([f"{t.capitalize()}={task_colors.get(t, ('gray', 'black'))[1]}" 
    #                       for t in task_combination])
    
    ax.set_xlabel('Low-rank Dimension 1', fontsize=7)
    ax.set_ylabel('Low-rank Dimension 2', fontsize=7)
    # ax.set_title(f'State Trajectories in Low-rank Space\n({task_str}: {color_str})', fontsize=7)
    ax.legend(fontsize=7, loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def main():
    """
    Main function to run the analysis.
    """

    
    # ============================================================================
    # CONFIGURATION - Modify these settings as needed
    # ============================================================================
    
    # Task combination: Choose any two tasks from ['flipflop', 'line', 'cycle']
    # Examples:
    #   ['flipflop', 'line']   - Flipflop and Line tasks
    #   ['flipflop', 'cycle']  - Flipflop and Cycle tasks
    #   ['line', 'cycle']      - Line and Cycle tasks
    task_combination = ['line', 'cycle']  # Change this to switch task combinations
    
    # Training mode: 'standard', 'orthogonal', or 'parallel'
    training_mode = "standard"  # Options: "standard", "orthogonal", "parallel"
    
    # Model path: Set to None for automatic search, or specify a path manually
    # Example: model_path = "models/flipfloporthogonal_2bits_200_rank2_100/i100/weights.pt"
    model_path = None  # Will search automatically if None
    
    units = 100
    n_bits = 2
    steps = 200
    n_trials_per_task = 100  # Number of trials per task type
    instance = 100
    
    # Set device (use CPU for inference to avoid device issues)
    pytorchtools.device = torch.device("cpu")
    
    # Validate task combination
    valid_tasks = ['flipflop', 'line', 'cycle']
    if len(task_combination) != 2 or not all(task in valid_tasks for task in task_combination):
        print(f"Error: task_combination must be a list of 2 tasks from {valid_tasks}")
        print(f"  Current: {task_combination}")
        return
    
    # Try to find model path
    if model_path is None:
        print(f"Searching for rank 2 model with tasks: {task_combination}, mode: {training_mode}...")
        model_path = find_model_path(task_combination=task_combination, training_mode=training_mode, rank=2, instance=instance)
        
        if model_path is None:
            print(f"\nError: Could not find a model for tasks {task_combination} with training mode '{training_mode}'.")
            print("Please specify the model path manually or check available models:")
            print("\nAvailable models in models/ directory:")
            models_dir = Path("models")
            if models_dir.exists():
                for folder in sorted(models_dir.iterdir()):
                    if folder.is_dir() and "rank2" in folder.name:
                        print(f"  - {folder.name}")
            print("\nTo use a specific model, set model_path in the script, e.g.:")
            print('  model_path = "models/flipfloporthogonal_2bits_200_rank2_100/i100/weights.pt"')
            print("\nOr change task_combination or training_mode")
            return
    
    print(f"\nLoading model from: {model_path}")
    print(f"Task combination: {task_combination}")
    print(f"Training mode: {training_mode}")
    
    # Load model and data generator
    model, data_generator, weights = load_model_and_data(
        model_path, units=units, n_bits=n_bits, steps=steps, 
        task_combination=task_combination, training_mode=training_mode
    )
    
    print(f"Model loaded successfully!")
    print(f"  Units: {units}")
    print(f"  Inputs: {data_generator.n_inputs}")
    print(f"  Outputs: {data_generator.n_outputs}")
    print(f"  Steps: {steps}")
    
    # Extract low-rank basis (U and V)
    print("\nExtracting low-rank basis from model...")
    V, U = extract_low_rank_basis(model, weights)
    print(f"  V (low-rank basis) shape: {V.shape}")
    print(f"  U shape: {U.shape}")
    
    # Run trials (ensures equal number for each task type)
    states, inputs, task_labels = run_trials(
        model, data_generator, task_combination=task_combination, 
        n_trials_per_task=n_trials_per_task, training_mode=training_mode
    )
    
    print(f"\nCollected states shape: {states.shape}")
    print(f"Task distribution:")
    for task in task_combination:
        print(f"  {task.capitalize()}: {task_labels.count(task)}")
    
    # Plot trajectories in low-rank space with task-based coloring
    task_str = '_'.join(sorted(task_combination))
    output_path = f"trajectories_low_rank_{task_str}_{training_mode}.png"
    plot_low_rank_trajectories(states, task_labels, V, task_combination, save_path=output_path)


if __name__ == "__main__":
    main()

