"""
Debug script for testing the ActivationAnalysisLoader outside of Streamlit.

Run this script to check if the data loader can find and access your analysis data.
"""

import os
import sys
import argparse
from data_loader import ActivationAnalysisLoader

def main():
    """Test the data loader with various paths."""
    parser = argparse.ArgumentParser(description='Debug the ActivationAnalysisLoader')
    parser.add_argument('--analysis-dir', type=str, default=None,
                       help='Path to the analysis directory')
    args = parser.parse_args()
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Testing data loader with analysis_dir: {args.analysis_dir}")
    
    try:
        loader = ActivationAnalysisLoader(args.analysis_dir)
        
        print(f"\nFound {len(loader.sweeps)} sweeps:")
        for sweep in loader.sweeps:
            print(f"  - {sweep}")
            runs = loader.get_runs_in_sweep(sweep)
            print(f"    Found {len(runs)} runs")
            if runs:
                print(f"    First run: {runs[0]}")
                try:
                    metadata = loader.load_metadata(sweep, runs[0])
                    print(f"    Metadata loaded successfully")
                except Exception as e:
                    print(f"    Error loading metadata: {e}")
                
                try:
                    regression_df = loader.load_regression_results(sweep, runs[0])
                    print(f"    Regression results loaded: {len(regression_df)} rows")
                except Exception as e:
                    print(f"    Error loading regression results: {e}")
    
    except Exception as e:
        print(f"Error initializing data loader: {e}")
    
    # Try some alternative paths to see if they work
    alternative_paths = [
        "analysis",
        "../analysis",
        os.path.join(os.path.dirname(os.getcwd()), "analysis")
    ]
    
    print("\nTrying alternative paths:")
    for path in alternative_paths:
        if os.path.exists(path):
            print(f"\nPath '{path}' exists. Contents:")
            try:
                contents = os.listdir(path)
                for item in contents:
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        print(f"  - DIR: {item}")
                    else:
                        print(f"  - FILE: {item}")
            except Exception as e:
                print(f"  Error listing directory: {e}")
        else:
            print(f"Path '{path}' does not exist")

if __name__ == "__main__":
    main() 