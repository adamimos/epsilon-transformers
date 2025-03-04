# Activation Analysis with Google Drive Support

This directory contains scripts for running activation analysis with Google Drive integration, particularly useful when running in Google Colab environments.

## Main Script: `main_drive.py`

`main_drive.py` is a modified version of the main activation analysis pipeline that supports loading models from Google Drive while using local MSP data storage.

### Prerequisites

- Google Colab environment with Google Drive mounted
- Epsilon Transformers package installed
- Models saved in a Google Drive directory

### Basic Usage in Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone the repository (if needed)
!git clone -b quantum-save https://github.com/adamimos/epsilon-transformers.git
%cd epsilon-transformers

# Install the package and dependencies
!pip install -e .

# Run the analysis script
!python scripts/activation_analysis/main_drive.py --drive-path="/content/drive/My Drive/quantum/"
```

### Command Line Arguments

- `--drive-path`: Path to the Google Drive directory containing models (required)
- `--sweep-id`: Process only runs from a specific sweep (optional)
- `--run-id`: Process only a specific run (optional)
- `--device`: Device to use (cpu, cuda, mps) (default: cpu)
- `--output-dir`: Custom output directory (default: {drive-path}/analysis)
- `--layer-filter`: Filter runs by layer pattern (default: 'L4')
- `--use-local-msp`: Use local MSP data storage (default: True)

### Example: Process a Specific Sweep

```bash
python scripts/activation_analysis/main_drive.py --drive-path="/content/drive/My Drive/quantum/" --sweep-id="transformer_sweep_v1"
```

### Example: Process a Specific Run with GPU

```bash
python scripts/activation_analysis/main_drive.py --drive-path="/content/drive/My Drive/quantum/" --sweep-id="transformer_sweep_v1" --run-id="run_L4_H128" --device="cuda"
```

### How It Works

1. The script initializes a `GoogleDriveModelLoader` with the specified drive path
2. It detects available sweeps in the drive path
3. For each sweep, it:
   - Infers the model type (transformer or RNN)
   - Lists available runs (filtered by layer pattern)
   - Processes each run:
     - Extracts activations from model checkpoints
     - Runs regression analysis on the activations
     - Saves results to the output directory
4. MSP data is stored locally to improve performance

### Notes

- This script requires Google Drive to be mounted when run in Google Colab
- The `--use-local-msp` flag ensures MSP data is stored locally rather than in Google Drive
- The script automatically detects and processes all available sweeps unless a specific sweep is specified 