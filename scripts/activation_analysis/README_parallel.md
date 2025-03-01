# Parallel Activation Analysis

This document explains how to use the `main_parallel.py` script for processing activation analysis tasks in parallel across multiple GPUs.

## Overview

The `main_parallel.py` script provides a flexible way to run activation analysis on specific runs or automatically distribute multiple runs across available GPUs. It's designed to:

1. Process a single run on a specific GPU
2. Auto-detect available runs and distribute them across GPUs
3. Distribute runs from a provided list across multiple GPUs
4. Save results to local directories or directly to S3 storage

## Basic Usage

### Process a Single Run

To process a single run on a specific GPU:

```bash
python -m scripts.activation_analysis.main_parallel --run-id YOUR_RUN_ID --gpu-id 0
```

### Auto-detect and Process Multiple Runs

To automatically detect runs and distribute them across multiple GPUs:

```bash
python -m scripts.activation_analysis.main_parallel --auto-detect --distribute --num-gpus 4
```

### Process a List of Runs

To process a specific list of runs:

```bash
python -m scripts.activation_analysis.main_parallel --run-list path/to/run_list.json --distribute --num-gpus 4
```

Where `run_list.json` is a JSON file containing a list of run IDs:

```json
[
  "run_id_1",
  "run_id_2",
  "run_id_3",
  ...
]
```

### Save Results to S3

To save results directly to an S3 bucket:

```bash
python -m scripts.activation_analysis.main_parallel --run-id YOUR_RUN_ID --s3-output s3://your-bucket/output/path
```

You can also use the main.py script directly with S3 storage:

```bash
python -m scripts.activation_analysis.main --run-id YOUR_RUN_ID --s3-output s3://your-bucket/output/path
```

## Command-Line Options

### Run Selection Options

- `--run-id RUNID`: Process a specific run ID
- `--auto-detect`: Automatically detect available runs to process
- `--run-list FILENAME`: Process run IDs from a JSON file

### Distribution Options

- `--distribute`: Enable distribution across multiple GPUs
- `--num-gpus N`: Number of GPUs to distribute jobs across (default: 1)
- `--gpu-id N`: GPU ID to use when processing a single run (default: 0)

### Processing Options

- `--sweep-id SWEEPID`: Process only runs from this sweep (defaults to auto-detect)
- `--output-dir DIR`: Output directory (defaults to value in config)
- `--s3-output PATH`: S3 output path (e.g., s3://bucket-name/path/to/output)
- `--log-dir DIR`: Log directory (defaults to OUTPUT_DIR/logs)
- `--max-checkpoints N`: Maximum number of checkpoints to process
- `--process-all-checkpoints`: Process all checkpoints, not just the most recent ones

### Execution Control

- `--dry-run`: Print commands without executing them
- `--wait`: Wait for each batch to complete before starting the next

## Examples

### Test a Single Run

```bash
python -m scripts.activation_analysis.main_parallel --run-id my_test_run --gpu-id 0
```

### Distribute 20 Runs Across 4 GPUs

```bash
python -m scripts.activation_analysis.main_parallel --run-list my_runs.json --distribute --num-gpus 4
```

### Auto-detect All Runs in a Specific Sweep

```bash
python -m scripts.activation_analysis.main_parallel --auto-detect --sweep-id my_sweep --distribute --num-gpus 2
```

### Save Results Directly to S3

```bash
python -m scripts.activation_analysis.main_parallel --run-id my_test_run --s3-output s3://my-bucket/analysis
```

### Dry Run to Test Configuration

```bash
python -m scripts.activation_analysis.main_parallel --auto-detect --distribute --num-gpus 4 --dry-run
```

## Using S3 Storage

Both `main.py` and `main_parallel.py` now support saving results directly to Amazon S3 storage. This feature requires:

1. AWS credentials configured either via environment variables or the AWS CLI
2. The boto3 Python package (installed via pip)

### Setting Up S3 Access

1. Install the AWS SDK for Python:
   ```bash
   pip install boto3
   ```

2. Configure your AWS credentials using one of these methods:
   
   a. Environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=your_region
   ```
   
   b. AWS CLI configuration:
   ```bash
   aws configure
   ```

3. Create a bucket in S3 if you don't already have one:
   ```bash
   aws s3 mb s3://your-bucket-name
   ```

### Using S3 Output

To save analysis results directly to S3, use the `--s3-output` flag:

```bash
python -m scripts.activation_analysis.main --s3-output s3://your-bucket/output/path
```

The script will automatically:
1. Detect that the path is an S3 URL
2. Save all output files (CSV, JSON, etc.) directly to the specified S3 path
3. Maintain the same directory structure in S3 as it would locally

### S3 Path Format

S3 paths should be in the format `s3://bucket-name/path/to/directory`. The script will verify and fix the format if necessary.

### Benefits of Using S3

- **Persistence**: Results are stored in the cloud and won't be lost if the compute instance terminates
- **Accessibility**: Access your data from anywhere without having to transfer files manually
- **Scalability**: S3 can handle large amounts of data without space constraints
- **Integration**: Easily integrate with other AWS services or analysis tools

## Tips for Running on Remote Systems

When running on cloud or cluster systems:

1. Use a tmux/screen session to prevent disconnects from terminating your jobs
2. Use the `--wait` flag when concerned about GPU memory usage
3. Limit the number of checkpoints with `--max-checkpoints` to reduce processing time
4. Test with `--dry-run` first to make sure everything is configured correctly
5. Check logs in the output directory to debug any issues
6. When using S3, keep your AWS credentials secure and consider using IAM roles

## Creating a Run List

You can create a JSON file with run IDs to process specific runs:

```json
[
  "run_id_1",
  "run_id_2",
  "run_id_3"
]
```

This approach lets you prepare lists of runs for different processing tasks or distribute them across different compute resources.

## Files

- `main.py`: Original script that processes all available runs
- `main_parallel.py`: A version of the main script that processes a single run_id on a specific GPU
- `main_parallel_simple.py`: A simplified version that combines the functionality of multiple scripts
- `example_run_list.json`: An example file showing the format for specifying runs to process

## Using with vast.ai

### Preparation

1. Clone your repository to your local machine
2. Generate the vast.ai execution scripts:

```bash
python -m scripts.activation_analysis.distribute_jobs \
  --num-gpus 4 \
  --generate-vast-ai \
  --process-all-checkpoints
```

This will create scripts in the `analysis/vast_ai_scripts/` directory.

3. (Optional) Edit the generated scripts to update the repository URL and any other settings

### Running on vast.ai

1. Create instances on vast.ai with the appropriate hardware (CUDA-compatible GPUs)
2. Upload the generated scripts to each instance (via SSH or the vast.ai interface)
3. Execute the appropriate script on each instance

For example, on instance 1:
```bash
bash run_gpu_0.sh
```

On instance 2:
```bash
bash run_gpu_1.sh
```

And so on.

### Monitoring

- Check the log files in the `logs` directory to monitor progress
- You can SSH into the vast.ai instances to check status or view logs in real-time

## Local Parallel Execution

You can also run multiple jobs in parallel on a local multi-GPU machine:

```bash
python -m scripts.activation_analysis.distribute_jobs \
  --num-gpus 2 \
  --wait
```

This will distribute jobs across 2 GPUs and wait for all jobs to complete.

## Command-Line Options

### main.py

```
usage: main.py [-h] [--output-dir OUTPUT_DIR] [--s3-output S3_OUTPUT]
               [--sweep-id SWEEP_ID] [--run-id RUN_ID] [--device DEVICE]

Run activation analysis pipeline

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Output directory (default: analysis)
  --s3-output S3_OUTPUT
                        S3 output path (e.g., s3://bucket-name/path/to/output)
  --sweep-id SWEEP_ID  Process only runs from this sweep (default: all sweeps)
  --run-id RUN_ID      Process only this specific run
  --device DEVICE      Device to use (default: cuda:0)
```

### main_parallel.py

```
usage: main_parallel.py [-h] --run-id RUN_ID [--sweep-id SWEEP_ID]
                        [--gpu-id GPU_ID] [--output-dir OUTPUT_DIR]
                        [--s3-output S3_OUTPUT] [--log-dir LOG_DIR] 
                        [--max-checkpoints MAX_CHECKPOINTS]
                        [--process-all-checkpoints]

Run activation analysis on specific runs

optional arguments:
  -h, --help            show this help message and exit
  --run-id RUN_ID       Run ID to process
  --sweep-id SWEEP_ID   Sweep ID to process (default: auto-detect from SWEEPS 
                        config)
  --gpu-id GPU_ID       GPU ID to use (default: 0)
  --output-dir OUTPUT_DIR
                        Output directory (default: analysis)
  --s3-output S3_OUTPUT
                        S3 output path (e.g., s3://bucket-name/path/to/output)
  --log-dir LOG_DIR     Log directory (defaults to OUTPUT_DIR/logs)
  --max-checkpoints MAX_CHECKPOINTS
                        Maximum number of checkpoints to process
  --process-all-checkpoints
                        Process all checkpoints, not just the most recent ones
```

## Tips for vast.ai

1. **Choose the right instance type**: For GPU-intensive tasks, look for instances with at least 16GB GPU memory
2. **Data persistence**: Use S3 for output to ensure data is preserved even if the instance goes down
3. **Cost management**: Monitor your usage and shut down instances when not needed
4. **Checkpoint frequently**: Save intermediate results to avoid losing progress if an instance fails 