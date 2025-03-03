# Activation Analysis Dashboard

This dashboard visualizes results from the activation analysis pipeline that investigates how neural networks represent belief states across different model architectures and tasks.

## Data Structure

The analysis output is organized in a hierarchical directory structure as follows:

```
analysis/
├── [sweep_id]/                # e.g., 20241121152808
│   ├── [run_id]/              # e.g., run_71_L4_H64_RNN_uni_mess3
│   │   ├── [run_id]_regression_results.csv
│   │   ├── [run_id]_random_baseline.csv
│   │   ├── [run_id]_loss.csv
│   │   └── [run_id]_metadata.json
│   └── ...
└── ...
```

### File Descriptions

#### 1. `[run_id]_metadata.json`

This JSON file contains metadata about the analysis run:

- **creation_time**: Timestamp when the analysis was performed
- **format_version**: Format version of the output files (e.g., "csv_v1")
- **run_id**: Identifier for the specific run
- **sweep_id**: Identifier for the sweep that contains this run
- **file_index**: Paths to all related files for this run

#### 2. `[run_id]_regression_results.csv`

This CSV file contains the main regression analysis results for each layer at different checkpoints:

| Column | Description |
|--------|-------------|
| layer_name | Name of the neural network layer being analyzed |
| layer_idx | Index of the layer in the network architecture |
| norm_dist | Normalized distance metric for the layer's representations |
| dims | Dimensionality of the layer (number of units/neurons) |
| r_squared | R-squared value indicating how well the layer represents the target |
| target | Target for regression (e.g., "nn_beliefs") |
| rcond | Reciprocal condition number used in regression calculations |
| checkpoint | Training checkpoint number (0 for initial state) |

This file captures how well each layer in the neural network can encode belief states at different points during training.

#### 3. `[run_id]_random_baseline.csv`

This CSV file contains regression results for random weight baselines:

| Column | Description |
|--------|-------------|
| layer_name | Name of the neural network layer |
| layer_idx | Index of the layer |
| norm_dist | Normalized distance metric |
| dims | Dimensionality of the layer |
| r_squared | R-squared value for random weights |
| checkpoint | Shows "RANDOM_X" where X is the random seed/index |
| target | Target for regression |
| rcond | Reciprocal condition number |

This provides a comparison baseline to distinguish meaningful representations from chance.

#### 4. `[run_id]_loss.csv`

This CSV file contains training and validation loss metrics over time:

| Column | Description |
|--------|-------------|
| epoch | Training epoch number |
| num_tokens_seen | Number of tokens processed during training |
| train_loss_ctx[N] | Training loss for context window N |
| val_loss_ctx[N] | Validation loss for context window N |
| train_loss_mean | Mean training loss across all context windows |
| val_loss_mean | Mean validation loss across all context windows |
| learning_rate | Learning rate used during training |

This tracks the model's learning progress over time for different context lengths.

### Naming Conventions

The run IDs follow a structured naming pattern that encodes important model configuration information:
- Example: `run_71_L4_H64_RNN_uni_mess3`
  - `run_71`: Run identifier number
  - `L4`: 4 layers in the network
  - `H64`: 64 hidden units per layer
  - `RNN`: Model architecture type (can also be LSTM, GRU, or transformer-style in other runs)
  - `uni`: Likely indicates unidirectional processing
  - `mess3`: Task or dataset name

Similarly, the transformer-style models use different naming:
- Example: `run_23_L4_H4_DH16_DM64_mess3`
  - `L4`: 4 layers
  - `H4`: 4 attention heads
  - `DH16`: Head dimension of 16
  - `DM64`: Model dimension of 64
  - `mess3`: Task name 