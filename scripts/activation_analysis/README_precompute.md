# Markov Approximation Precomputation

This script precomputes Markov approximations for various process configurations and saves them for later use. By precomputing these approximations, we can avoid redundant computations when multiple runs share the same process parameters.

## Features

- Precompute Markov approximations for one or more process configurations
- Save the results in a format that can be loaded by the activation analysis pipeline
- Automatically stop processing higher-order Markov approximations when the belief dimension reaches 64
- Support for different ways to specify process configurations (via config file, command line arguments, or existing sweeps)

## Usage

### Precompute from a Config File

Create a JSON file with process configurations (see `example_process_configs.json`), then run:

```bash
python -m scripts.activation_analysis.precompute_markov --config scripts/activation_analysis/example_process_configs.json
```

### Precompute for a Specific Process Configuration

Specify a process configuration directly as a JSON string:

```bash
python -m scripts.activation_analysis.precompute_markov --process-config '{"type": "even_odd", "n_states": 8}'
```

Or using separate arguments:

```bash
python -m scripts.activation_analysis.precompute_markov --process-type even_odd --n-states 8
```

### Precompute for Existing Sweeps

Process all sweeps defined in the configuration:

```bash
python -m scripts.activation_analysis.precompute_markov --all-sweeps
```

Or a specific sweep:

```bash
python -m scripts.activation_analysis.precompute_markov --sweep-id SWEEP_ID
```

### Additional Options

- `--max-order`: Maximum Markov order to compute (default: 4)
- `--device`: Device to use (default: same as in config.py)

## Example Config File

```json
{
  "model_config": {
    "n_ctx": 8,
    "n_embd": 64,
    "n_layer": 4
  },
  "process_configs": [
    {
      "type": "even_odd",
      "n_states": 8
    },
    {
      "type": "parity",
      "n_states": 8
    },
    {
      "type": "random",
      "n_states": 16,
      "sparsity": 0.7,
      "seed": 123
    }
  ]
}
```

## Benefits

1. **Performance Optimization**: Avoid redundant computations when multiple runs use the same process parameters.
2. **Consistency**: Ensure all runs use exactly the same Markov approximations.
3. **Resource Management**: Precompute intensive calculations during off-peak hours or on more powerful machines.

## Implementation Notes

- Results are saved using the same caching mechanism as the main activation analysis pipeline.
- The script intelligently checks for existing cached data and skips configurations that have already been processed.
- Higher-order Markov approximations are automatically skipped when the belief dimension reaches or exceeds 64. 