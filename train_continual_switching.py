import numpy as np
from train_continual import ContinualLearningTrainer, set_training_device
from model.model_wrapper import ModelWrapper, OptimizationParameters
from data.data_generator import FamilyOfTasksGenerator, RNNDataGenerator
from data.custom_data_generator import CustomDataGenerator
from plot_continual_learning_loss import plot_continual_learning_loss


class SwitchingContinualLearningTrainer(ContinualLearningTrainer):
    
    def __init__(self, architecture_func, units, tasks_list, active_task_indices, 
                 instance_range, optimization_params=None, device='auto', **kwargs):
        """
        Args:
            architecture_func: RNN architecture class
            units: Number of hidden units
            tasks_list: List of task lists, each list contains ALL tasks for that stage
                        e.g., [[TaskA, TaskB], [TaskA, TaskB], [TaskA, TaskB]]
            active_task_indices: List of task indices to train in each stage
                                e.g., [0, 1, 0] means:
                                - Stage 0: train task 0 (TaskA)
                                - Stage 1: train task 1 (TaskB)
                                - Stage 2: train task 0 (TaskA)
            instance_range: Range of model instances to train
            optimization_params: Training parameters
            device: Device to use ('mps', 'cpu', or 'auto')
            **kwargs: Additional arguments for ModelWrapper
        """

        if len(tasks_list) != len(active_task_indices):
            raise ValueError(f"tasks_list length ({len(tasks_list)}) must match "
                           f"active_task_indices length ({len(active_task_indices)})")
        
        n_tasks_per_stage = len(tasks_list[0])
        for i, tasks in enumerate(tasks_list):
            if len(tasks) != n_tasks_per_stage:
                raise ValueError(f"All stages must have the same number of tasks. "
                               f"Stage {i} has {len(tasks)} tasks, expected {n_tasks_per_stage}")
        
        for i, task_idx in enumerate(active_task_indices):
            if task_idx < 0 or task_idx >= n_tasks_per_stage:
                raise ValueError(f"Stage {i}: active_task_indices[{i}]={task_idx} "
                               f"is out of range [0, {n_tasks_per_stage-1}]")
        
        self.active_task_indices = active_task_indices
        self.n_tasks_total = n_tasks_per_stage
        
        super().__init__(
            architecture_func=architecture_func,
            units=units,
            tasks_list=tasks_list,
            instance_range=instance_range,
            optimization_params=optimization_params,
            device=device,
            n_tasks_total=n_tasks_per_stage,
            **kwargs
        )
    
    def train_stage(self, stage_idx, tasks, initial_weights=None):
        """
        Train on a specific stage, but only on the active task.
        
        Args:
            stage_idx: Index of the current learning stage
            tasks: List of ALL tasks for this stage
            initial_weights: Weights to initialize from (from previous stage)
        """
        active_task_idx = self.active_task_indices[stage_idx]
        active_task = tasks[active_task_idx]
        
        print(f"\n{'='*60}")
        print(f"Training Stage {stage_idx + 1}/{len(self.tasks_list)}")
        print(f"All tasks: {[task.name for task in tasks]}")
        print(f"Active task (training): {active_task.name} (index {active_task_idx})")
        print(f"{'='*60}\n")
        
        is_custom_generator = isinstance(active_task, CustomDataGenerator)
        
        if is_custom_generator:
            train_data = active_task
            
            if train_data.n_inputs != self.n_tasks_total or train_data.n_outputs != self.n_tasks_total:
                class DimensionAdjustedGenerator(CustomDataGenerator):
                    def __init__(self, base_generator, n_tasks_total):
                        super().__init__(
                            n_inputs=n_tasks_total,
                            n_outputs=n_tasks_total,
                            n_batch=base_generator.n_batch,
                            steps=base_generator.steps,
                            min_delay=base_generator.min_delay,
                            max_delay=base_generator.max_delay,
                            delta=base_generator.delta,
                            vmin=base_generator.vmin,
                            vmax=base_generator.vmax
                        )
                        self.base_generator = base_generator
                        self.active_task_idx = active_task_idx
                    
                    @property
                    def task_name(self):
                        return self.base_generator.task_name
                    
                    def generate_training_trial(self):
                        if hasattr(self.base_generator, 'generate_trial'):
                            x_seq = self.base_generator.generate_random_sequence()
                            x_base, y_base = self.base_generator.generate_trial(0, x_seq)
                            
                            x = np.zeros((self.steps, self.n_inputs))
                            y = np.empty((self.steps, self.n_outputs))
                            y[:] = np.nan
                            
                            x[:, self.active_task_idx] = x_base[:, 0]
                            y[:, self.active_task_idx] = y_base[:, 0]
                        else:
                            x_base, y_base = self.base_generator.generate_training_trial()
                            
                            x = np.zeros((self.steps, self.n_inputs))
                            y = np.empty((self.steps, self.n_outputs))
                            y[:] = np.nan
                            
                            if x_base.shape[1] >= 1:
                                x[:, self.active_task_idx] = x_base[:, 0]
                            if y_base.shape[1] >= 1:
                                y[:, self.active_task_idx] = y_base[:, 0]
                        
                        return x, y
                    
                    def generate_validation_data(self):
                        if hasattr(self.base_generator, 'generate_trial'):
                            reps = 100
                            n_trials = reps
                            x = np.zeros((n_trials, self.steps, self.n_inputs))
                            y = np.empty((n_trials, self.steps, self.n_outputs))
                            y[:] = np.nan
                            
                            for i in range(n_trials):
                                x_seq = self.base_generator.generate_random_sequence(val=True)
                                x_base, y_base = self.base_generator.generate_trial(0, x_seq)
                                x[i, :, self.active_task_idx] = x_base[:, 0]
                                y[i, :, self.active_task_idx] = y_base[:, 0]
                            
                            return x, y
                        else:
                            x_base, y_base = self.base_generator.generate_validation_data()
                            
                            n_trials = x_base.shape[0]
                            x = np.zeros((n_trials, self.steps, self.n_inputs))
                            y = np.empty((n_trials, self.steps, self.n_outputs))
                            y[:] = np.nan
                            
                            for i in range(n_trials):
                                if x_base.shape[2] >= 1:
                                    x[i, :, self.active_task_idx] = x_base[i, :, 0]
                                if y_base.shape[2] >= 1:
                                    y[i, :, self.active_task_idx] = y_base[i, :, 0]
                            
                            return x, y
                
                train_data = DimensionAdjustedGenerator(active_task, self.n_tasks_total)
            else:
                original_generate_training_trial = train_data.generate_training_trial
                
                def generate_active_task_trial():
                    x_base, y_base = original_generate_training_trial()
                    x = np.zeros_like(x_base)
                    y = np.empty_like(y_base)
                    y[:] = np.nan
                    
                    if x_base.shape[1] > 0:
                        x[:, active_task_idx] = x_base[:, 0] if x_base.shape[1] == 1 else x_base[:, active_task_idx]
                    if y_base.shape[1] > 0:
                        y[:, active_task_idx] = y_base[:, 0] if y_base.shape[1] == 1 else y_base[:, active_task_idx]
                    
                    return x, y
                
                train_data.generate_training_trial = generate_active_task_trial
        else:
            train_data = FamilyOfTasksGenerator(
                tasks,
                input_type=self.kwargs.get('input_type', 'multi'),
                output_type=self.kwargs.get('output_type', 'multiple'),
                steps=self.kwargs.get('steps', 250),
                n_tasks_total=self.n_tasks_total,
                train_last=1
            )
            
            original_generate_trial = train_data.generate_trial
            
            def generate_active_task_trial():
                x_seq = train_data.generate_random_sequence()
                xx = train_data.construct_input(active_task_idx, x_seq)
                yy = train_data.generate_task_output(xx, active_task_idx)
                return xx, yy
            
            train_data.generate_training_trial = generate_active_task_trial
            
            original_generate_validation = train_data.generate_validation_data
            
            def generate_active_task_validation():
                reps = 100
                n_trials = reps
                x = np.zeros((n_trials, train_data.steps, train_data.n_inputs))
                y = np.zeros((n_trials, train_data.steps, train_data.n_outputs))
                
                for rep in range(reps):
                    x_seq = train_data.generate_random_sequence(val=True)
                    xx = train_data.construct_input(active_task_idx, x_seq)
                    yy = train_data.generate_task_output(xx, active_task_idx)
                    x[rep] = xx
                    y[rep] = yy
                
                return x, y
            
            train_data.generate_validation_data = generate_active_task_validation
        
        wrapper = ModelWrapper(
            architecture_func=self.architecture_func,
            units=self.units,
            train_data=train_data,
            instance_range=self.instance_range,
            optimization_params=self.optimization_params,
            **{k: v for k, v in self.kwargs.items() 
               if k not in ['input_type', 'output_type', 'steps', 'n_tasks_total']}
        )
        
        weights = None
        if initial_weights is not None:
            weights = initial_weights
        
        # Train
        wrapper.train_model(weights=weights)
        
        # Save weights for next stage
        self.stage_weights[stage_idx] = wrapper.get_all_weights()
        self.stage_wrappers.append(wrapper)
        
        # Print model save location
        print(f"\nStage {stage_idx + 1} models saved to:")
        for inst in self.instance_range:
            model_path = f"models/{wrapper.name}/i{inst}"
            print(f"  Instance {inst}: {model_path}/weights.pt")
        
        return wrapper



def switching_continual_learning_example(device='auto', plot_loss=True, instance=None):

    from data.functions import X, X2
    from model.pt_models import VanillaArchitecture, Rank2Architechture
    from data.custom_data_generator import OrthogonalFlipFlopGenerator, OrthogonalCyclesGenerator, FlipFlopGenerator, LinesGenerator
    
    TaskA = FlipFlopGenerator(n_bits=2)
    TaskB = LinesGenerator(n_bits=2)
    
    stages = [
        [TaskA, TaskB],
        [TaskA, TaskB],
    ]
    
    active_task_indices = [1, 0]
    
    trainer = SwitchingContinualLearningTrainer(
        architecture_func=Rank2Architechture,
        units=100,
        tasks_list=stages,
        active_task_indices=active_task_indices,
        instance_range=range(1220, 1221),
        optimization_params=OptimizationParameters(
            batch_size=32,
            epochs=500,
            minimal_loss=1e-4,
            initial_lr=1e-4
        ),
        device=device,
        recurrent_bias=False,
        readout_bias=True
    )
    
    wrappers = trainer.train_all_stages()
    
    # Get instance from instance_range if not provided
    if instance is None:
        instance = list(trainer.instance_range)[0] if trainer.instance_range else 0
    
    # Plot training loss curves if requested
    if plot_loss:
        print("\n" + "="*60)
        print("Plotting training loss curves...")
        print("="*60)
        try:
            plot_continual_learning_loss(
                trainer=trainer,
                instance=instance,
                save_path=f'continual_learning_loss_instance_{instance}.png'
            )
        except Exception as e:
            print(f"Warning: Could not plot loss curves: {e}")
            import traceback
            traceback.print_exc()
            print("\nYou can try plotting manually:")
            print(f"  from plot_continual_learning_loss import plot_continual_learning_loss")
            print(f"  plot_continual_learning_loss(trainer, instance={instance})")
    
    return trainer, wrappers


if __name__ == '__main__':
    import numpy as np
    trainer, wrappers = switching_continual_learning_example(device='cpu')

