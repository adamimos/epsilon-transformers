#!/usr/bin/env python
"""
Simple script to test the S3ModelLoader with both default and company credentials.
This script prints out the available sweeps from both S3 buckets and attempts to list runs in the first sweep.
"""
from epsilon_transformers.analysis.load_data import S3ModelLoader
import os
import sys
from dotenv import load_dotenv

def test_s3_loader(use_company_credentials=False):
    """Test the S3ModelLoader with the specified credentials."""
    loader_type = "COMPANY" if use_company_credentials else "DEFAULT"
    print(f"\nTesting S3ModelLoader with {loader_type} credentials...")
    
    try:
        # Initialize S3ModelLoader
        loader = S3ModelLoader(use_company_credentials=use_company_credentials)
        print(f"Bucket name: {loader.bucket_name}")
        print(f"Path prefix: '{loader.path_prefix}'")
        
        # List sweeps
        sweeps = loader.list_sweeps()
        print(f"\nFound {len(sweeps)} sweeps in bucket:")
        for sweep in sweeps[:10]:  # Print first 10 to avoid overwhelming output
            print(f"  - {sweep}")
        if len(sweeps) > 10:
            print(f"  ... and {len(sweeps) - 10} more")
            
        # Test deeper access by listing runs in the first sweep
        if sweeps:
            test_sweep = sweeps[0]
            print(f"\nTesting run listing for sweep: {test_sweep}")
            runs = loader.list_runs_in_sweep(test_sweep)
            print(f"Found {len(runs)} runs in sweep {test_sweep}:")
            for run in runs[:5]:  # Print first 5 runs
                print(f"  - {run}")
            if len(runs) > 5:
                print(f"  ... and {len(runs) - 5} more")
                
            # Test even deeper access by listing checkpoints for the first run
            if runs:
                test_run = runs[0]
                print(f"\nTesting checkpoint listing for run: {test_run}")
                checkpoints = loader.list_checkpoints(test_sweep, test_run)
                print(f"Found {len(checkpoints)} checkpoints in run {test_run}:")
                for ckpt in checkpoints[:3]:  # Print first 3 checkpoints
                    print(f"  - {ckpt}")
                if len(checkpoints) > 3:
                    print(f"  ... and {len(checkpoints) - 3} more")
        else:
            print("\nNo sweeps found to test run listing.")
            
        return True
            
    except Exception as e:
        print(f"Error with {loader_type.lower()} loader: {str(e)}")
        return False

def main():
    load_dotenv()
    
    # Print environment variables for debugging (without exposing secrets)
    print("Environment variables check:")
    print(f"AWS_DEFAULT_REGION set: {'Yes' if os.getenv('AWS_DEFAULT_REGION') else 'No'}")
    print(f"S3_BUCKET_NAME set: {'Yes' if os.getenv('S3_BUCKET_NAME') else 'No'}")
    print(f"COMPANY_AWS_DEFAULT_REGION set: {'Yes' if os.getenv('COMPANY_AWS_DEFAULT_REGION') else 'No'}")
    print(f"COMPANY_S3_BUCKET_NAME set: {'Yes' if os.getenv('COMPANY_S3_BUCKET_NAME') else 'No'}")
    
    # Test default S3 loader
    default_success = test_s3_loader(use_company_credentials=False)
    
    print("\n" + "-" * 70 + "\n")
    
    # Test company S3 loader
    company_success = test_s3_loader(use_company_credentials=True)
    
    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY:")
    print(f"Default S3 access: {'SUCCESS' if default_success else 'FAILED'}")
    print(f"Company S3 access: {'SUCCESS' if company_success else 'FAILED'}")
    
    # Return error code if either test failed
    if not (default_success and company_success):
        sys.exit(1)

if __name__ == "__main__":
    main() 