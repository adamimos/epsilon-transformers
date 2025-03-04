#!/bin/bash

# Define script for better error handling
set -e

# Print information about the run
echo "==================== STARTING ALL EXPERIMENTS ===================="
echo "Running experiments with different models and learning rates"
echo "Timestamp: $(date)"
echo ""

# Run the transformer experiments
echo "==================== TRANSFORMER LR=1e-2 ===================="
python ./scripts/launcher_cuda_parallel.py --config ./scripts/experiment_config_tom_quantum_short_lr_1e-2.yaml
echo "Transformer experiment with learning rate 1e-2 completed"
echo ""

echo "==================== TRANSFORMER LR=1e-4 ===================="
python ./scripts/launcher_cuda_parallel.py --config ./scripts/experiment_config_tom_quantum_short_lr_1e-4.yaml
echo "Transformer experiment with learning rate 1e-4 completed"
echo ""

# Run the RNN experiments
echo "==================== RNN LR=1e-2 ===================="
python ./scripts/launcher_cuda_parallel_rnn.py --config ./scripts/experiment_config_tom_quantum_rnn_short_lr_1e-2.yaml
echo "RNN experiment with learning rate 1e-2 completed"
echo ""

echo "==================== RNN LR=1e-4 ===================="
python ./scripts/launcher_cuda_parallel_rnn.py --config ./scripts/experiment_config_tom_quantum_rnn_short_lr_1e-4.yaml
echo "RNN experiment with learning rate 1e-4 completed"
echo ""

echo "==================== ALL EXPERIMENTS COMPLETED ===================="
echo "Timestamp: $(date)" 