"""
Script to analyze pseudoinverse comparison results.

This script loads and analyzes the CSV results from compare_pinv_methods.py,
providing detailed insights into performance differences and accuracy.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def analyze_comparison_results(csv_file, output_dir=None):
    """
    Analyze pseudoinverse comparison results from CSV file.
    
    Args:
        csv_file: Path to the comparison results CSV file
        output_dir: Directory to save visualization files (default: same as CSV)
    """
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_file))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading comparison results from {csv_file}")
    results = pd.read_csv(csv_file)
    
    # Basic statistics
    print("\n=== BASIC STATISTICS ===")
    print(f"Total rows: {len(results)}")
    print(f"Unique layers: {results['layer_name'].nunique()}")
    print(f"Unique rcond values: {results['rcond'].nunique()}")
    print(f"Unique targets: {results['target'].nunique()}")
    
    # Performance statistics
    print("\n=== PERFORMANCE STATISTICS ===")
    speedup = results['speedup_factor'].iloc[0]  # Speedup is the same for all rows
    print(f"Speedup factor: {speedup:.2f}x")
    print(f"This means the optimized implementation is {(speedup - 1) * 100:.1f}% faster")
    
    # Accuracy statistics
    print("\n=== ACCURACY STATISTICS ===")
    print(f"Mean absolute relative difference: {results['norm_dist_rel_diff'].mean():.8e}")
    print(f"Max absolute relative difference: {results['norm_dist_rel_diff'].max():.8e}")
    print(f"Mean absolute r-squared difference: {results['r_squared_diff'].abs().mean():.8e}")
    
    # Group by layer, rcond, and target
    layer_stats = results.groupby('layer_name').agg({
        'norm_dist_rel_diff': ['mean', 'max'],
        'r_squared_diff': lambda x: np.abs(x).mean()
    }).reset_index()
    
    rcond_stats = results.groupby('rcond').agg({
        'norm_dist_rel_diff': ['mean', 'max'],
        'r_squared_diff': lambda x: np.abs(x).mean()
    }).reset_index()
    
    target_stats = results.groupby('target').agg({
        'norm_dist_rel_diff': ['mean', 'max'],
        'r_squared_diff': lambda x: np.abs(x).mean()
    }).reset_index()
    
    # Generate PDF report with visualizations
    pdf_path = os.path.join(output_dir, 'pinv_comparison_analysis.pdf')
    with PdfPages(pdf_path) as pdf:
        # Create title page
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.9, 'Pseudoinverse Comparison Analysis', 
                horizontalalignment='center', fontsize=20)
        plt.text(0.5, 0.85, f'Report for: {os.path.basename(csv_file)}', 
                horizontalalignment='center')
        plt.text(0.5, 0.8, f'Speedup Factor: {speedup:.2f}x', 
                horizontalalignment='center', fontsize=16)
        plt.text(0.5, 0.7, f'Mean Relative Error: {results["norm_dist_rel_diff"].mean():.2e}', 
                horizontalalignment='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Plot layer statistics
        plt.figure(figsize=(10, 6))
        plt.bar(layer_stats['layer_name'], layer_stats[('norm_dist_rel_diff', 'mean')])
        plt.title('Mean Relative Difference by Layer')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Plot rcond statistics
        plt.figure(figsize=(8, 6))
        plt.plot(rcond_stats['rcond'], rcond_stats[('norm_dist_rel_diff', 'mean')], 'o-')
        plt.title('Mean Relative Difference by rcond Value')
        plt.xlabel('rcond Value')
        plt.ylabel('Mean Relative Difference')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Plot r-squared differences
        plt.figure(figsize=(10, 6))
        plt.bar(layer_stats['layer_name'], layer_stats[('r_squared_diff', 'lambda')])
        plt.title('Mean Absolute R-squared Difference by Layer')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
    print(f"\nAnalysis complete. Report saved to {pdf_path}")
    return results

def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(description='Analyze pseudoinverse comparison results')
    parser.add_argument('csv_file', type=str, help='Path to the comparison results CSV file')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Directory to save analysis outputs')
    args = parser.parse_args()
    
    analyze_comparison_results(args.csv_file, args.output_dir)

if __name__ == "__main__":
    main() 