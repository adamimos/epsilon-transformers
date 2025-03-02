"""
Configuration settings for the activation analysis pipeline.
All configuration parameters are centralized in this file.
"""
import torch
import os
import logging
from datetime import datetime

# ======================================================
# Global configuration and defaults
# ======================================================

class AnalysisConfig:
    """Centralized configuration class for the analysis pipeline."""
    
    # Output and logging
    OUTPUT_DIR = "analysis"
    # Logging level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    # DEBUG provides the most detailed logging, while CRITICAL provides the least
    LOG_LEVEL = logging.WARNING  # Default level
    # Uncomment one of these lines for different logging levels:
    # LOG_LEVEL = logging.DEBUG  # For detailed debugging information
    # LOG_LEVEL = logging.WARNING  # For warnings and errors only
    # LOG_LEVEL = logging.ERROR  # For errors only
    # LOG_LEVEL = logging.CRITICAL  # For critical errors only
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = os.path.join(LOG_DIR, f"analysis_{TIMESTAMP}.log")
    
    # Analysis features
    REPORT_VARIANCE = True
    DO_BASELINE = True
    NUM_RANDOM_BASELINES = 1000  # Set to 10000 for full analysis
    
    # Device configuration
    @staticmethod
    def get_default_device():
        """Determine the best available device for computation."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    
    # Default device - can be overridden if needed
    # DEFAULT_DEVICE = 'cpu'  # Uncomment to force CPU usage
    DEFAULT_DEVICE = get_default_device()
    
    # Regression configuration
    RCOND_SWEEP_LIST = [1e-15, 1e-10, 1e-5] + list(torch.logspace(-8, -3, 50).tolist())
    
    # Sweeps to process
    SWEEPS = {
        '20241205175736': 'transformer',
        '20241121152808': 'rnn'
    }
    
    # Markov analysis settings
    MAX_MARKOV_ORDER = 4
    
    # Checkpoint processing settings
    PROCESS_ALL_CHECKPOINTS = True  # Set to True to process all checkpoints
    MAX_CHECKPOINTS = 200  # Only process this many checkpoints if PROCESS_ALL_CHECKPOINTS is False
    
    # Layer configurations
    TRANSFORMER_ACTIVATION_KEYS = (
        ['blocks.0.hook_resid_pre'] +
        [f'blocks.{i}.hook_resid_post' for i in range(10)] +
        ['ln_final.hook_normalized']
    )
    
    # Memory management
    MEMORY_EFFICIENT = True  # When True, aggressively release memory when possible
    MAX_CACHE_SIZE_GB = 4.0  # Maximum size of cached data in GB
    
    # Error handling
    MAX_RETRIES = 3  # Maximum number of retries for S3 operations
    RETRY_DELAY = 5  # Delay between retries in seconds

# Create a global instance for easy access
CONFIG = AnalysisConfig()

# Make old-style imports work by setting module-level variables
# This allows existing code to still work with minimal changes
OUTPUT_DIR = CONFIG.OUTPUT_DIR
REPORT_VARIANCE = CONFIG.REPORT_VARIANCE
DO_BASELINE = CONFIG.DO_BASELINE
NUM_RANDOM_BASELINES = CONFIG.NUM_RANDOM_BASELINES
DEFAULT_DEVICE = CONFIG.DEFAULT_DEVICE
RCOND_SWEEP_LIST = CONFIG.RCOND_SWEEP_LIST
SWEEPS = CONFIG.SWEEPS
MAX_MARKOV_ORDER = CONFIG.MAX_MARKOV_ORDER
PROCESS_ALL_CHECKPOINTS = CONFIG.PROCESS_ALL_CHECKPOINTS
MAX_CHECKPOINTS = CONFIG.MAX_CHECKPOINTS
TRANSFORMER_ACTIVATION_KEYS = CONFIG.TRANSFORMER_ACTIVATION_KEYS

# Import convenience - allows importing directly from config
from scripts.activation_analysis.utils import setup_logging

# Automatically set up logging when this module is imported
os.makedirs(CONFIG.LOG_DIR, exist_ok=True)
setup_logging(log_dir=CONFIG.LOG_DIR, log_level=CONFIG.LOG_LEVEL)