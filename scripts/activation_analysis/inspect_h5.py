#!/usr/bin/env python
"""
Script to inspect H5 files created by the activation analysis pipeline.
Run with: python inspect_h5.py path/to/your/file.h5
"""

import h5py
import numpy as np
import pandas as pd
import argparse
import os
from pprint import pprint
import sys

def list_attrs(item, indent=0):
    """List all attributes of an H5 item."""
    if len(item.attrs.keys()) > 0:
        print(" " * indent + "Attributes:")
        for key, val in item.attrs.items():
            print(" " * (indent + 2) + f"{key}: {val}")

def explore_h5(file_path, detailed=False, summary=False, group_path=None):
    """
    Explore the content of an H5 file.
    
    Args:
        file_path: Path to the H5 file
        detailed: Whether to show detailed information
        summary: Show only a summary of the file
        group_path: Optional path to a specific group to explore
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    print(f"\nExploring H5 file: {file_path}\n")
    
    with h5py.File(file_path, 'r') as f:
        if summary:
            print_file_summary(f)
        elif group_path:
            if group_path in f:
                explore_group(f[group_path], detailed=detailed, indent=0)
            else:
                print(f"Group {group_path} not found in file.")
                available_groups = list(f.keys())
                print(f"Available top-level groups: {available_groups}")
        else:
            # Print overall structure first
            print("File structure:")
            f.visit(lambda name: print(f"  {name}"))
            print("\nDetailed content:")
            for key in f.keys():
                explore_group(f[key], detailed=detailed, indent=0)

def explore_group(group, detailed=False, indent=0):
    """Recursively explore an H5 group."""
    indent_str = " " * indent
    print(f"{indent_str}Group: {group.name}")
    list_attrs(group, indent)
    
    # Process sub-groups
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            explore_group(item, detailed, indent + 2)
        else:  # dataset
            explore_dataset(item, detailed, indent + 2)

def explore_dataset(dataset, detailed=False, indent=0):
    """Explore an H5 dataset."""
    indent_str = " " * indent
    shape_str = f"shape={dataset.shape}" if hasattr(dataset, 'shape') else ""
    dtype_str = f"dtype={dataset.dtype}" if hasattr(dataset, 'dtype') else ""
    print(f"{indent_str}Dataset: {dataset.name} ({shape_str}, {dtype_str})")
    list_attrs(dataset, indent)
    
    if detailed:
        if len(dataset.shape) == 0 or dataset.shape[0] <= 10:
            try:
                data = dataset[:]
                print(f"{indent_str}  Data: {data}")
            except Exception as e:
                print(f"{indent_str}  Error reading data: {e}")
        else:
            try:
                # Show first few elements
                data = dataset[:10]
                print(f"{indent_str}  First 10 elements: {data}")
            except Exception as e:
                print(f"{indent_str}  Error reading data: {e}")

def print_file_summary(h5file):
    """Print a summary of the H5 file contents."""
    # Count total number of groups, datasets, and get total size
    num_groups = 0
    num_datasets = 0
    total_size = 0

    def visitor(name, obj):
        nonlocal num_groups, num_datasets, total_size
        if isinstance(obj, h5py.Group):
            num_groups += 1
        elif isinstance(obj, h5py.Dataset):
            num_datasets += 1
            total_size += obj.size * obj.dtype.itemsize

    h5file.visititems(visitor)
    
    # Print summary
    print("=== File Summary ===")
    print(f"Total number of groups: {num_groups}")
    print(f"Total number of datasets: {num_datasets}")
    print(f"Total data size: {total_size / 1024 / 1024:.2f} MB")
    print("\nTop-level groups:")
    for group_name in h5file.keys():
        print(f"  {group_name}")
        
    # Show dataframe if "regression_results" exists
    if "regression_results" in h5file:
        print("\nRegression Results Summary:")
        df = load_regression_df(h5file)
        print_df_summary(df)
        
    # Check if weights and singular values exist
    if "best_weights" in h5file:
        print("\nBest Weights Structure:")
        explore_weights_summary(h5file["best_weights"])
    
    if "singular_values" in h5file:
        print("\nSingular Values Structure:")
        explore_singular_values_summary(h5file["singular_values"])

def explore_weights_summary(weights_group):
    """Print a summary of the weights structure."""
    targets = list(weights_group.keys())
    print(f"Number of targets: {len(targets)}")
    print(f"Targets: {targets}")
    
    # Check a sample target
    if targets:
        sample_target = targets[0]
        layers = list(weights_group[sample_target].keys())
        print(f"\nSample target '{sample_target}' has {len(layers)} layers")
        print(f"Layers: {layers}")
        
        # Check a sample layer
        if layers:
            sample_layer = layers[0]
            print(f"\nSample layer '{sample_layer}' data:")
            layer_group = weights_group[sample_target][sample_layer]
            
            # List datasets and attributes
            datasets = list(layer_group.keys())
            print(f"  Datasets: {datasets}")
            print(f"  Attributes: {list(layer_group.attrs.keys())}")
            
            # Print shape of weights if available
            if "weights" in layer_group:
                weights = layer_group["weights"]
                print(f"  Weights shape: {weights.shape}, dtype: {weights.dtype}")

def explore_singular_values_summary(sv_group):
    """Print a summary of the singular values structure."""
    targets = list(sv_group.keys())
    print(f"Number of targets: {len(targets)}")
    print(f"Targets: {targets}")
    
    # Check a sample target
    if targets:
        sample_target = targets[0]
        layers = list(sv_group[sample_target].keys())
        print(f"\nSample target '{sample_target}' has {len(layers)} layers")
        print(f"Layers: {layers}")
        
        # Check a sample layer
        if layers:
            sample_layer = layers[0]
            entries = list(sv_group[sample_target][sample_layer].keys())
            print(f"\nSample layer '{sample_layer}' has {len(entries)} entries")
            
            # Check a sample entry
            if entries:
                sample_entry = entries[0]
                entry_group = sv_group[sample_target][sample_layer][sample_entry]
                
                # List datasets and attributes
                datasets = list(entry_group.keys())
                print(f"  Datasets: {datasets}")
                print(f"  Attributes: {list(entry_group.attrs.keys())}")
                
                # Print shape of singular values if available
                if "singular_values" in entry_group:
                    sv = entry_group["singular_values"]
                    print(f"  Singular values shape: {sv.shape}, dtype: {sv.dtype}")

def load_regression_df(h5file):
    """Load regression results into a pandas DataFrame."""
    results_group = h5file["regression_results"]
    data = {}
    
    for col in results_group.keys():
        data[col] = results_group[col][:]
        
        # Convert bytes to strings if needed
        if isinstance(data[col][0], bytes):
            data[col] = [x.decode('utf-8') for x in data[col]]
    
    return pd.DataFrame(data)

def print_df_summary(df):
    """Print a summary of the regression results DataFrame."""
    print(f"DataFrame shape: {df.shape}")
    print("\nColumns:", ", ".join(df.columns))
    
    if 'target' in df.columns:
        targets = df['target'].unique()
        print(f"\nUnique targets ({len(targets)}):", ", ".join(targets))
    
    if 'layer_name' in df.columns:
        layers = df['layer_name'].unique()
        print(f"\nUnique layers ({len(layers)}):", ", ".join(layers))
    
    if 'rcond' in df.columns:
        rcond_values = df['rcond'].unique()
        print(f"\nNumber of rcond values: {len(rcond_values)}")
        print(f"Range: {min(rcond_values)} to {max(rcond_values)}")
    
    if 'checkpoint' in df.columns:
        checkpoints = df['checkpoint'].unique()
        print(f"\nUnique checkpoints ({len(checkpoints)}):", ", ".join(map(str, checkpoints)))
    
    print("\nSample data (5 rows):")
    print(df.head())

def main():
    parser = argparse.ArgumentParser(description='Inspect H5 files from the activation analysis pipeline.')
    parser.add_argument('file_path', help='Path to the H5 file')
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed information including data values')
    parser.add_argument('--summary', '-s', action='store_true', help='Show only a summary of the file')
    parser.add_argument('--group', '-g', help='Explore a specific group path within the file')
    parser.add_argument('--extract', '-e', help='Extract regression DataFrame to CSV')
    
    args = parser.parse_args()
    
    if args.extract:
        extract_regression_df(args.file_path, args.extract)
    else:
        explore_h5(args.file_path, detailed=args.detailed, summary=args.summary, group_path=args.group)

def extract_regression_df(file_path, output_path):
    """Extract the regression results DataFrame to a CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    try:
        with h5py.File(file_path, 'r') as f:
            if "regression_results" not in f:
                print("Error: No regression_results group found in the file.")
                return
            
            df = load_regression_df(f)
            df.to_csv(output_path, index=False)
            print(f"Successfully extracted regression results to {output_path}")
            print(f"DataFrame shape: {df.shape}")
    except Exception as e:
        print(f"Error extracting regression DataFrame: {e}")

if __name__ == "__main__":
    main()