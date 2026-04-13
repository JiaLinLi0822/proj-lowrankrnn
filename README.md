# Multitask Representation Learning in Recurrent Neural Networks

A computational neuroscience research project investigating how recurrent neural networks (RNNs) learn and represent multiple tasks. This repository contains tools for training RNNs on various task families, analyzing their internal representations, and studying the relationship between task structure and learned representations.

## Overview

This project explores how RNNs develop internal representations when trained on multiple tasks simultaneously. It includes:

- **Multiple RNN Architectures**: Vanilla RNNs, rank-constrained RNNs (Rank1, Rank2, Rank3, Rank50), LSTM, GRU, and multi-layer variants
- **Diverse Task Generators**: Flip-flop tasks, function families (polynomial, trigonometric), and custom memory tasks
- **Comprehensive Analysis Tools**: PCA, spectrum analysis, kappa plane visualization, basin of attraction analysis, and task similarity metrics
- **Training Infrastructure**: PyTorch-based training with checkpointing and validation

## Project Structure

```
multitaskrepresentation-main/
├── analysis/              # Analysis modules for studying representations
│   ├── analyzer.py        # Base analyzer class
│   ├── spectrum_analysis.py
│   ├── kappaplane.py      # Low-rank representation analysis
│   ├── tasks_pca.py       # PCA analysis of task representations
│   ├── basinofattraction.py
│   ├── taskpairwisekappadistance.py
│   └── ...
├── data/                  # Data generation modules
│   ├── data_generator.py  # Base data generator and task families
│   ├── custom_data_generator.py  # Flip-flop and memory tasks
│   ├── functions.py       # Mathematical function definitions
│   └── data_config.py     # Task configuration presets
├── model/                 # Model definitions and training
│   ├── pt_models.py       # PyTorch RNN architectures
│   ├── model_wrapper.py   # High-level model interface
│   ├── trainer.py         # Training loop and loss functions
│   └── rnn_models.py      # Base RNN model class
├── tools/                 # Utility functions
│   ├── pytorchtools.py   # PyTorch utilities
│   ├── training_utils.py
│   └── math_utils.py
├── models/                # Trained model weights (generated)
├── analysis_results/      # Analysis outputs (generated)
├── figures/               # Generated figures
└── *.py                   # Main analysis scripts
```

## Key Features

### RNN Architectures

The project supports various RNN architectures:

- **VanillaArchitecture**: Standard fully-connected RNN
- **Rank2Architechture**: Low-rank RNN with rank-2 recurrent weight matrix (m @ n^T)
- **Rank1Architechture, Rank3Architechture, Rank50Architechture**: Other rank-constrained variants
- **LSTMArchitecture, GRUArchitecture**: Gated RNN variants
- **GRUMultiArchitecture**: Multi-layer RNN

### Task Types

#### Function Family Tasks
- Polynomial functions: `X()`, `X2()`, `X4()`, etc.
- Rotated and reversed variants: `X2Rotate()`, `XReverse()`, etc.
- Trigonometric functions: `Sine()`, `CoSine()`, `Tanh()`
- 2D functions: `L1()`, `L2()`, `LMAX()`, etc.

#### Memory Tasks
- **FlipFlopGenerator**: Binary flip-flop memory task
- **ParallelFlipFlopGenerator**: Parallel flip-flop tasks
- **OrthogonalFlipFlopGenerator**: Orthogonal flip-flop tasks
- **CyclesGenerator**: Oscillatory output tasks
- **LinesGenerator**: Continuous value memory tasks

### Analysis Tools

1. **SpectrumAnalysis**: Analyzes eigenvalue spectrum of recurrent weight matrix
2. **FFPCA**: PCA analysis of flip-flop task representations
3. **TasksPCA**: PCA analysis across multiple tasks
4. **KappaPlane**: Visualization of low-rank representation space
5. **BasinOfAttraction**: Studies attractor dynamics
6. **TasksPairwiseKappaDistance**: Measures task similarity in representation space

## Installation

### Prerequisites

This project requires:
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- SciPy

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multitaskrepresentation-main
```

2. Set up a conda environment (recommended):
```bash
conda create -n deeprl python=3.x
conda activate deeprl
```

3. Install dependencies:
```bash
pip install torch numpy matplotlib scikit-learn scipy seaborn cached-property
```

## Usage

### Basic Training and Analysis

The project includes several example scripts:

#### 1. Flip-Flop Spectrum Analysis (`bits_to_outliers.py`)

Analyzes the eigenvalue spectrum of RNNs trained on flip-flop tasks:

```python
from model.model_wrapper import ModelWrapper
from model.pt_models import VanillaArchitecture
from data.custom_data_generator import FlipFlopGenerator
from analysis.spectrum_analysis import SpectrumAnalysis

kwargs = {
    'architecture_func': VanillaArchitecture,
    'units': 100,
    'train_data': FlipFlopGenerator(n_bits=2),
    'instance_range': range(800, 810),
    'recurrent_bias': False,
    'readout_bias': True
}

wrapper = ModelWrapper(**kwargs)
wrapper.train_model()
wrapper.analyze([SpectrumAnalysis])
```

#### 2. Task Family PCA Analysis (`plot_ff_pca.py`)

Performs PCA analysis on representations learned for different tasks:

```python
from analysis.tasks_pca import TasksPCA
from data.data_generator import FamilyOfTasksGenerator
from data.functions import X2, X2Rotate

tasks = [X2(), X2Rotate()]
train_data = FamilyOfTasksGenerator(tasks, steps=200)

wrapper = ModelWrapper(
    architecture_func=VanillaArchitecture,
    units=100,
    train_data=train_data,
    instance_range=[5000]
)
wrapper.train_model()
wrapper.analyze([TasksPCA])
```

#### 3. Low-Rank Representation Analysis (`plot_kappa_6tasks.py`)

Visualizes the kappa plane for rank-2 RNNs trained on multiple tasks:

```python
from analysis.kappaplane import KappaPlane
from model.pt_models import Rank2Architechture
from data.functions import *

TASKS = [X4Rotate(), X2Rotate(), XReverse(), X(), X2(), X4()]

wrapper = ModelWrapper(
    architecture_func=Rank2Architechture,
    units=100,
    train_data=FamilyOfTasksGenerator(TASKS),
    instance_range=range(2000, 2020)
)
wrapper.train_model()
wrapper.analyze([KappaPlane])
```

#### 4. Task Similarity Analysis (`task_and_representation_distance.py`)

Measures the relationship between task function similarity and representation distance:

```python
from analysis.taskpairwisekappadistance import TasksPairwiseKappaDistance

TASKS = [X4Rotate(), X2Rotate(), XReverse(), X(), X2(), X4()]

wrapper = ModelWrapper(
    architecture_func=Rank2Architechture,
    units=100,
    train_data=FamilyOfTasksGenerator(TASKS),
    instance_range=range(2000, 2025)
)
wrapper.train_model()
wrapper.analyze([TasksPairwiseKappaDistance])
```

### ModelWrapper API

The `ModelWrapper` class provides a high-level interface for training and analysis:

```python
# Initialize wrapper
wrapper = ModelWrapper(
    architecture_func=VanillaArchitecture,  # Architecture type
    units=100,                               # Hidden units
    train_data=train_data,                   # Data generator
    instance_range=range(0, 10),            # Model instances to train
    recurrent_bias=False,                    # Recurrent layer bias
    readout_bias=True,                       # Output layer bias
    optimization_params=OptimizationParameters(...)  # Training params
)

# Train models
wrapper.train_model()

# Run analyses
wrapper.analyze([SpectrumAnalysis, TasksPCA])

# Get analysis results
results = wrapper.get_file('soft_average')  # Load saved analysis
```

### Data Generators

#### FamilyOfTasksGenerator

Generates sequences for multiple tasks:

```python
from data.data_generator import FamilyOfTasksGenerator
from data.functions import X2, X2Rotate, X4

tasks = [X2(), X2Rotate(), X4()]
generator = FamilyOfTasksGenerator(
    tasks,
    input_type='multi',      # 'multi', 'single', 'tonic', 'transient'
    output_type='multiple',  # 'single', 'multiple', 'all'
    steps=250,
    n_values_training=15
)
```

#### Custom Memory Tasks

```python
from data.custom_data_generator import (
    FlipFlopGenerator,
    ParallelFlipFlopGenerator,
    OrthogonalFlipFlopGenerator
)

# Binary flip-flop task
ff_task = FlipFlopGenerator(n_bits=2)

# Parallel tasks (all bits active simultaneously)
parallel_ff = ParallelFlipFlopGenerator(n_bits=2)

# Orthogonal tasks (separate input channels)
orthogonal_ff = OrthogonalFlipFlopGenerator(n_bits=2)
```

## Output Structure

### Model Storage

Trained models are saved in `models/`:
```
models/
└── {task_name}_{architecture_name}/
    └── i{instance}/
        ├── weights.pt              # Final weights
        ├── initial_weights.pt      # Initial weights
        └── {analysis_name}.pkl     # Analysis results
```

### Analysis Results

Analysis outputs are saved in `analysis_results/`:
```
analysis_results/
└── {analysis_name}/
    └── {description}_{model_name}_i{instance}.png
```

## Continual Learning

The project supports continual learning where tasks are learned sequentially. This is useful for studying:
- How RNNs adapt to new tasks while retaining knowledge of previous tasks
- Catastrophic forgetting and methods to prevent it
- Transfer learning between related tasks

### Basic Continual Learning

```python
from train_continual import ContinualLearningTrainer
from data.functions import X, X2, X2Rotate, X4

# Define learning stages - tasks are added incrementally
stages = [
    [X()],                          # Stage 1: Learn X
    [X(), X2()],                    # Stage 2: Add X2
    [X(), X2(), X2Rotate()],       # Stage 3: Add X2Rotate
    [X(), X2(), X2Rotate(), X4()], # Stage 4: Add X4
]

trainer = ContinualLearningTrainer(
    architecture_func=VanillaArchitecture,
    units=100,
    tasks_list=stages,
    instance_range=range(0, 5)
)

# Train all stages sequentially
wrappers = trainer.train_all_stages()
```

### Continual Learning with Replay

To prevent catastrophic forgetting, use the replay version that mixes data from previous tasks:

```python
from train_continual import ContinualLearningWithReplay

trainer = ContinualLearningWithReplay(
    architecture_func=VanillaArchitecture,
    units=100,
    tasks_list=stages,
    instance_range=range(0, 3),
    replay_ratio=0.3  # 30% of training data from previous tasks
)

wrappers = trainer.train_all_stages()
```

### Running Continual Learning

```bash
# Basic continual learning
python train_continual.py 1

# With replay buffer
python train_continual.py 2

# Rank2 architecture
python train_continual.py 3
```

## Key Scripts

- `bits_to_outliers.py`: Spectrum analysis for flip-flop tasks
- `plot_ff_pca.py`: PCA analysis of flip-flop representations
- `plot_kappa_6tasks.py`: Kappa plane visualization for 6 tasks
- `plotrank2learning.py`: Learning dynamics in rank-2 RNNs
- `task_and_representation_distance.py`: Task similarity analysis
- `plot_architectures.py`: Architecture comparison
- `train_continual.py`: Continual learning training script

## Configuration

### Optimization Parameters

```python
from model.model_wrapper import OptimizationParameters

params = OptimizationParameters(
    batch_size=32,
    loss='mse',
    epochs=10000,
    minimal_loss=1e-4,
    initial_lr=1e-4
)
```

### Architecture Parameters

- `units`: Number of hidden units
- `recurrent_bias`: Whether to use bias in recurrent layer
- `readout_bias`: Whether to use bias in output layer
- `activation`: Activation function ('tanh', 'relu', etc.)
- `freeze_params`: List of parameter names to freeze during training

## Dependencies

Core dependencies:
- `torch`: PyTorch for neural network implementation
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `scikit-learn`: Machine learning utilities (PCA, etc.)
- `scipy`: Scientific computing
- `seaborn`: Statistical visualization
- `cached-property`: Property caching decorator

## Notes

- The project uses a conda environment named 'deeprl' (as configured)
- Models are saved with instance numbers for reproducibility
- Analysis results are automatically saved as pickle files and plots
- Training continues until loss falls below `minimal_loss` or `epochs` is reached

## Citation

If you use this code in your research, please cite appropriately.

## License

[Specify license if applicable]
