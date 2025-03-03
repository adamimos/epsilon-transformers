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
    
    # Find where the maximum difference occurs
    max_diff_idx = results['norm_dist_rel_diff'].idxmax()
    max_diff_row = results.loc[max_diff_idx]
    print(f"\n=== MAXIMUM DIFFERENCE DETAILS ===")
    print(f"Layer: {max_diff_row['layer_name']}")
    print(f"Target: {max_diff_row['target']}")
    print(f"Rcond value: {max_diff_row['rcond']:.8e}")
    print(f"Original norm distance: {max_diff_row['norm_dist_orig']:.8e}")
    print(f"Optimized norm distance: {max_diff_row['norm_dist_opt']:.8e}")
    print(f"Absolute difference: {abs(max_diff_row['norm_dist_diff']):.8e}")
    print(f"Relative difference: {max_diff_row['norm_dist_rel_diff']:.8e}")
    print(f"Original R²: {max_diff_row['r_squared_orig']:.8f}")
    print(f"Optimized R²: {max_diff_row['r_squared_opt']:.8f}")
    
    # Find minimum distances across rcond values for each layer/target
    print("\n=== MINIMUM DISTANCE ANALYSIS ===")
    # Group by layer, target and find minimum norm_dist for each implementation
    min_dist = results.groupby(['layer_name', 'target']).agg({
        'norm_dist_orig': 'min',
        'norm_dist_opt': 'min',
        'rcond': 'count'  # Just to count how many rcond values were tested
    }).reset_index()
    
    # Calculate relative difference in minimum distances
    min_dist['min_dist_rel_diff'] = abs(min_dist['norm_dist_orig'] - min_dist['norm_dist_opt']) / min_dist['norm_dist_orig']
    
    # Print summary
    print(f"Mean relative difference in minimum distances: {min_dist['min_dist_rel_diff'].mean():.8e}")
    print(f"Max relative difference in minimum distances: {min_dist['min_dist_rel_diff'].max():.8e}")
    
    # Print the layer/target with the largest discrepancy in minimum distance
    max_min_diff_idx = min_dist['min_dist_rel_diff'].idxmax()
    max_min_diff_row = min_dist.loc[max_min_diff_idx]
    print(f"\nLargest discrepancy in minimum distance:")
    print(f"Layer: {max_min_diff_row['layer_name']}")
    print(f"Target: {max_min_diff_row['target']}")
    print(f"Original minimum: {max_min_diff_row['norm_dist_orig']:.8e}")
    print(f"Optimized minimum: {max_min_diff_row['norm_dist_opt']:.8e}")
    print(f"Relative difference: {max_min_diff_row['min_dist_rel_diff']:.8e}")
    
    # Group by layer, rcond, and target
    layer_stats = results.groupby('layer_name').agg({
        'norm_dist_rel_diff': ['mean', 'max'],
        'r_squared_diff': 'mean'
    }).reset_index()
    
    # Flatten column MultiIndex for easier access
    layer_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in layer_stats.columns]
    
    rcond_stats = results.groupby('rcond').agg({
        'norm_dist_rel_diff': ['mean', 'max'],
        'r_squared_diff': 'mean'
    }).reset_index()
    
    # Flatten column MultiIndex for easier access
    rcond_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in rcond_stats.columns]
    
    target_stats = results.groupby('target').agg({
        'norm_dist_rel_diff': ['mean', 'max'],
        'r_squared_diff': 'mean'
    }).reset_index()
    
    # Flatten column MultiIndex for easier access
    target_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in target_stats.columns]
    
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
        plt.bar(layer_stats['layer_name'], layer_stats['norm_dist_rel_diff_mean'])
        plt.title('Mean Relative Difference by Layer')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Plot rcond statistics
        plt.figure(figsize=(8, 6))
        plt.plot(rcond_stats['rcond'], rcond_stats['norm_dist_rel_diff_mean'], 'o-')
        plt.title('Mean Relative Difference by rcond Value')
        plt.xlabel('rcond Value')
        plt.ylabel('Mean Relative Difference')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Plot r-squared differences
        plt.figure(figsize=(10, 6))
        plt.bar(layer_stats['layer_name'], layer_stats['r_squared_diff_mean'])
        plt.title('Mean Absolute R-squared Difference by Layer')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Generate a histogram of relative differences
        plt.figure(figsize=(10, 6))
        plt.hist(results['norm_dist_rel_diff'], bins=50)
        plt.title('Distribution of Relative Differences in Norm Distance')
        plt.xlabel('Relative Difference')
        plt.ylabel('Count')
        plt.yscale('log')  # Use log scale to better see the distribution
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Plot minimum distance analysis
        plt.figure(figsize=(12, 6))
        plt.scatter(min_dist['layer_name'], min_dist['min_dist_rel_diff'], c=min_dist.index, cmap='viridis', s=100)
        plt.title('Relative Difference in Minimum Distances by Layer')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Relative Difference')
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Index')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Add a specific plot for the layer/target with largest minimum distance discrepancy
        max_diff_layer = max_min_diff_row['layer_name']
        max_diff_target = max_min_diff_row['target']
        
        # Filter data for this layer/target
        layer_target_data = results[(results['layer_name'] == max_diff_layer) & 
                                   (results['target'] == max_diff_target)]
        
        # Sort by rcond for proper line plotting
        layer_target_data = layer_target_data.sort_values('rcond')
        
        # Plot distances vs rcond for this layer/target
        plt.figure(figsize=(12, 8))
        
        # Plot on linear scale
        plt.subplot(2, 1, 1)
        plt.plot(layer_target_data['rcond'], layer_target_data['norm_dist_orig'], 'b-', label='Original')
        plt.plot(layer_target_data['rcond'], layer_target_data['norm_dist_opt'], 'r--', label='Optimized')
        plt.title(f'Distance vs rcond for {max_diff_layer} / {max_diff_target}')
        plt.xlabel('rcond value')
        plt.ylabel('Distance')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Also plot on log-log scale to better see differences
        plt.subplot(2, 1, 2)
        plt.loglog(layer_target_data['rcond'], layer_target_data['norm_dist_orig'], 'b-', label='Original')
        plt.loglog(layer_target_data['rcond'], layer_target_data['norm_dist_opt'], 'r--', label='Optimized')
        plt.title(f'Distance vs rcond (log-log scale) for {max_diff_layer} / {max_diff_target}')
        plt.xlabel('rcond value (log scale)')
        plt.ylabel('Distance (log scale)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Add another plot to show absolute and relative differences
        plt.figure(figsize=(12, 8))
        
        # Plot absolute difference
        plt.subplot(2, 1, 1)
        plt.plot(layer_target_data['rcond'], layer_target_data['norm_dist_diff'].abs())
        plt.title(f'Absolute Difference vs rcond for {max_diff_layer} / {max_diff_target}')
        plt.xlabel('rcond value')
        plt.ylabel('Absolute Difference')
        plt.grid(True, alpha=0.3)
        
        # Plot relative difference
        plt.subplot(2, 1, 2)
        plt.plot(layer_target_data['rcond'], layer_target_data['norm_dist_rel_diff'])
        plt.title(f'Relative Difference vs rcond for {max_diff_layer} / {max_diff_target}')
        plt.xlabel('rcond value')
        plt.ylabel('Relative Difference')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Create a section header page for individual plots
        plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.5, 'Individual Layer/Target Analysis', 
                horizontalalignment='center', fontsize=20)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Generate separate plots for each unique target and layer combination
        print("\n=== GENERATING INDIVIDUAL LAYER/TARGET PLOTS ===")
        # Get unique combinations of layer and target
        layer_target_combos = results[['layer_name', 'target']].drop_duplicates()
        print(f"Creating plots for {len(layer_target_combos)} layer/target combinations...")
        
        for _, row in layer_target_combos.iterrows():
            layer = row['layer_name']
            target = row['target']
            
            # Filter data for this layer/target
            combo_data = results[(results['layer_name'] == layer) & 
                                (results['target'] == target)]
            
            # Skip if no data (shouldn't happen, but just in case)
            if len(combo_data) == 0:
                continue
                
            # Sort by rcond for proper line plotting
            combo_data = combo_data.sort_values('rcond')
            
            # Calculate min distances and relative difference for this combo
            min_orig = combo_data['norm_dist_orig'].min()
            min_opt = combo_data['norm_dist_opt'].min()
            rel_diff = abs(min_orig - min_opt) / min_orig if min_orig > 0 else 0
            
            # Plot distances vs rcond for this layer/target
            plt.figure(figsize=(12, 8))
            
            # Add a title showing the min distance stats
            plt.suptitle(f"{layer} / {target}\nMin Dist Orig: {min_orig:.8e}, Min Dist Opt: {min_opt:.8e}\nRel Diff: {rel_diff:.8e}", 
                         fontsize=12, y=0.98)
            
            # Plot on linear scale
            plt.subplot(2, 1, 1)
            plt.plot(combo_data['rcond'], combo_data['norm_dist_orig'], 'b-', label='Original')
            plt.plot(combo_data['rcond'], combo_data['norm_dist_opt'], 'r--', label='Optimized')
            plt.title(f'Distance vs rcond')
            plt.xlabel('rcond value')
            plt.ylabel('Distance')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Also plot on log-log scale
            plt.subplot(2, 1, 2)
            plt.loglog(combo_data['rcond'], combo_data['norm_dist_orig'], 'b-', label='Original')
            plt.loglog(combo_data['rcond'], combo_data['norm_dist_opt'], 'r--', label='Optimized')
            plt.title(f'Distance vs rcond (log-log scale)')
            plt.xlabel('rcond value (log scale)')
            plt.ylabel('Distance (log scale)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout(rect=(0, 0, 1, 0.95))  # Adjust layout to accommodate suptitle
            pdf.savefig()
            plt.close()
            
            # Add a second plot showing the differences
            plt.figure(figsize=(12, 8))
            plt.suptitle(f"{layer} / {target} - Differences", fontsize=12, y=0.98)
            
            # Plot absolute difference
            plt.subplot(2, 1, 1)
            plt.plot(combo_data['rcond'], combo_data['norm_dist_diff'].abs())
            plt.title(f'Absolute Difference vs rcond')
            plt.xlabel('rcond value')
            plt.ylabel('Absolute Difference')
            plt.grid(True, alpha=0.3)
            
            # Plot relative difference
            plt.subplot(2, 1, 2)
            plt.plot(combo_data['rcond'], combo_data['norm_dist_rel_diff'])
            plt.title(f'Relative Difference vs rcond')
            plt.xlabel('rcond value')
            plt.ylabel('Relative Difference')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout(rect=(0, 0, 1, 0.95))
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