#!/bin/bash
# Simple script to run activation analysis from the repository root

# Ensure we're running from the repository root
if [ ! -d "scripts/activation_analysis" ]; then
    echo "Error: Please run this script from the repository root"
    exit 1
fi

# The first argument should be the type of analysis to run
ANALYSIS_TYPE=$1

# Remove the first argument
shift

# Default options
PYTHON_CMD=${PYTHON_CMD:-python}
DRY_RUN=false

# Parse command-line options
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --python=*)
            PYTHON_CMD="${arg#*=}"
            ;;
    esac
done

# Define default commands
run_main() {
    CMD="$PYTHON_CMD -m scripts.activation_analysis.main $@"
    if [ "$DRY_RUN" = true ]; then
        echo "Would run: $CMD"
    else
        echo "Running: $CMD"
        eval $CMD
    fi
}

run_parallel() {
    CMD="$PYTHON_CMD -m scripts.activation_analysis.main_parallel $@"
    if [ "$DRY_RUN" = true ]; then
        echo "Would run: $CMD"
    else
        echo "Running: $CMD"
        eval $CMD
    fi
}

# Decide what to do based on the analysis type
case $ANALYSIS_TYPE in
    main)
        # Run main.py with all args passed through
        run_main "$@"
        ;;
    parallel)
        # Run main_parallel.py with all args passed through
        run_parallel "$@"
        ;;
    single)
        # Example: Process a single run
        if [ -z "$1" ]; then
            echo "Error: Missing run ID"
            echo "Usage: $0 single RUN_ID [--sweep-id SWEEP_ID] [--gpu-id GPU_ID]"
            exit 1
        fi
        RUN_ID=$1
        shift
        run_parallel --run-id $RUN_ID "$@"
        ;;
    multi)
        # Example: Distribute multiple runs across GPUs
        NUM_GPUS=${1:-1}
        shift
        run_parallel --auto-detect --distribute --num-gpus $NUM_GPUS "$@"
        ;;
    s3)
        # Example: Save results to S3
        if [ -z "$1" ] || [ -z "$2" ]; then
            echo "Error: Missing run ID or S3 path"
            echo "Usage: $0 s3 RUN_ID S3_PATH [--gpu-id GPU_ID]"
            exit 1
        fi
        RUN_ID=$1
        S3_PATH=$2
        shift 2
        run_parallel --run-id $RUN_ID --s3-output $S3_PATH "$@"
        ;;
    help|--help|-h)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  main [args]                Run the main analysis script with all arguments passed through"
        echo "  parallel [args]            Run the parallel analysis script with all arguments passed through"
        echo "  single RUN_ID [args]       Process a single run with the given ID"
        echo "  multi NUM_GPUS [args]      Auto-detect runs and distribute across multiple GPUs"
        echo "  s3 RUN_ID S3_PATH [args]   Process a run and save results to S3"
        echo "  help                       Show this help message"
        echo ""
        echo "Options:"
        echo "  --dry-run                  Print commands without executing them"
        echo "  --python=PATH              Use a specific Python executable (default: python)"
        echo ""
        echo "Examples:"
        echo "  $0 single my_run_id --gpu-id 0"
        echo "  $0 multi 4 --process-all-checkpoints"
        echo "  $0 s3 my_run_id s3://my-bucket/output"
        echo "  $0 main --s3-output s3://my-bucket/output"
        echo "  $0 parallel --run-id my_run_id --gpu-id 1"
        ;;
    *)
        echo "Unknown command: $ANALYSIS_TYPE"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac 