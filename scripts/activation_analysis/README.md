# Activation Analysis: Data Structure Documentation

## Overview

The activation analysis pipeline extracts neural network activations, performs regression analysis to predict belief states, and stores the results in a structured format for further analysis and visualization. This document explains the data structure used by the system.

## Data Storage Format

The system uses a unified HDF5 file format that consolidates all results in a single file, including both trained network data and random baseline data.

Files are named: `<run_id>_unified_results.h5`

### Key Benefits

- Keeps all related data in a single file
- Simplifies data management and access
- Provides a clear organization between checkpoint (trained) and random baseline data
- Maintains all the benefits of hierarchical storage
- Optimized to avoid data redundancy

## Data Structure Details

### Unified H5 Format Structure

The unified H5 file contains two main groups and global file attributes:

```
<run_id>_unified_results.h5
├── attrs
│   ├── unified_format = True
│   ├── creation_time = "YYYY-MM-DD HH:MM:SS"
│   ├── run_id = "<run_id>"
│   └── sweep_id = "<sweep_id>"
├── checkpoint_results/           # Trained network results
│   ├── regression_results/       # DataFrame containing regression metrics
│   │   ├── layer_name
│   │   ├── target
│   │   ├── checkpoint
│   │   ├── rcond
│   │   ├── norm_dist
│   │   ├── r_squared
│   │   └── ...
│   ├── best_weights/             # Best regression weights per checkpoint
│   │   └── <target>/
│   │       └── <checkpoint>/
│   │           └── <layer>/
│   │               ├── weights       # Weight values
│   │               ├── attr: rcond
│   │               └── attr: dist
│   └── singular_values/          # SVD results
│       └── <target>/
│           └── <layer>/
│               └── entry_<idx>/
│                   ├── singular_values
│                   └── attr: checkpoint
└── random_results/               # Random network results
    ├── regression_results/       # DataFrame with same structure as above
    └── singular_values/          # SVD results for random networks
        └── <target>/
            └── <layer>/
                └── entry_<idx>/
                    ├── singular_values
                    └── attr: random_idx
```

### Regression Results Data

The regression results dataframe contains the following columns:

| Column      | Description |
|-------------|-------------|
| layer_name  | Name of the neural network layer |
| target      | Target belief state (e.g., 'nn_beliefs', 'markov_1', 'markov_2') |
| checkpoint  | Checkpoint identifier (or 'RANDOM_X' for random networks) |
| rcond       | Regularization parameter used in regression |
| norm_dist   | Normalized distance metric (lower is better) |
| r_squared   | R² coefficient of determination (higher is better) |
| is_random   | Flag indicating whether this is from a random baseline (True) or trained network (False) |

Note: Global metadata like `sweep_id` and `run_id` are stored as file attributes rather than redundantly in each row.

### Weight Data

The system stores the best weights for each checkpoint, target, and layer combination:

- The weights are organized hierarchically by target → checkpoint → layer
- For each layer, we store:
  - **weights**: The actual regression coefficients
  - **rcond**: The regularization parameter that produced the best result
  - **dist**: The achieved normalized distance

This approach ensures we keep the best weights from each checkpoint, rather than just a single best across all checkpoints.

### Singular Value Data

Singular value data captures the results of Singular Value Decomposition (SVD) performed on activations:

- **singular_values**: The singular values from SVD
- **checkpoint**: For trained networks, which checkpoint these values are from
- **random_idx**: For random networks, which random initialization these values are from

## Difference Between Random and Trained Network Data

The key distinction between random and trained network data:

- **Trained network data** (checkpoint results) depends on specific checkpoints during training, showing how the network evolves.
- **Random network data** represents baseline performance with randomly initialized weights, independent of checkpoints.

Both types of data share the same structure and metrics, allowing direct comparison in analysis.

## Storage Optimizations

The following optimizations have been implemented to ensure efficient data storage:

1. **Unified Format Only**: Only a single HDF5 file is saved per run, eliminating redundant storage
2. **Global Metadata**: Run ID and sweep ID are stored once as file attributes rather than in every row
3. **Checkpoint-Based Weights**: Best weights are organized by checkpoint for efficient access
4. **Compression**: All datasets use gzip compression (level 4) to reduce file size
5. **No Redundant Files**: No separate CSV files are generated, as all data is accessible from the H5 file

These optimizations significantly reduce storage requirements while maintaining all necessary information for analysis.

## Other Files

In addition to the unified H5 file, the system also generates:

- `<run_id>_loss.csv` - Training loss data if available (separate because it comes from the training process)

## Using the Data

### Dashboard Access

The interactive dashboard automatically handles the data format. Simply:

1. Specify the data directory
2. Click "Load Runs"
3. Select a run to visualize

### Programmatic Access

Example code to access the unified format:

```python
import h5py
import pandas as pd
import numpy as np

def load_unified_data(h5_file):
    with h5py.File(h5_file, 'r') as f:
        # Check if this is a unified format file
        if 'unified_format' not in f.attrs or not f.attrs['unified_format']:
            return None
            
        # Get global metadata
        run_id = f.attrs.get('run_id')
        sweep_id = f.attrs.get('sweep_id')
            
        # Load checkpoint results
        checkpoint_df = None
        if 'checkpoint_results/regression_results' in f:
            df_dict = {}
            for col in f['checkpoint_results/regression_results']:
                data = f['checkpoint_results/regression_results'][col][()]
                if data.dtype.kind == 'S':
                    data = np.array([s.decode('utf-8') for s in data])
                df_dict[col] = data
            checkpoint_df = pd.DataFrame(df_dict)
            
            # Add back metadata
            if run_id is not None:
                checkpoint_df['run_id'] = run_id
            if sweep_id is not None:
                checkpoint_df['sweep_id'] = sweep_id
            checkpoint_df['is_random'] = False
        
        return checkpoint_df
```

## Best Practices

- When analyzing programmatically, check for both checkpoint and random data
- Pay attention to the `is_random` flag when combining dataframes
- For visualization, use the dashboard which is optimized for this data format
- Remember that best weights are now stored per-checkpoint for more detailed analysis

The data structure is designed to make it easy to analyze how well neural network activations predict belief states, comparing across layers, regularization parameters, and between trained and random networks. 