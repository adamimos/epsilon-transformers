# Optimized Pseudoinverse Implementation

This document describes the optimized pseudoinverse implementation for regression analysis in the activation analysis pipeline. The optimization computes the Singular Value Decomposition (SVD) once and reuses it for multiple regularization condition (`rcond`) values, which can significantly improve performance when sweeping across multiple `rcond` values.

## Background

The original implementation computes a pseudoinverse (`pinv`) for each regularization condition (`rcond`) value independently, which involves repeatedly computing the SVD of the same matrix. This is computationally inefficient when analyzing the same data with multiple `rcond` values.

The optimized implementation:
1. Computes the SVD once: U, S, Vh = svd(A)
2. For each `rcond` value, creates a different Σ⁺ by thresholding singular values based on: threshold = `rcond` * max(singular_values)
3. Replaces singular values greater than the threshold with 1/value, and others with 0
4. Computes the pseudoinverse as V * Σ⁺ * U.T for each modified Σ⁺
5. Returns all pseudoinverses in a dictionary

## Implementation Details

The optimization is implemented in the `compute_efficient_pinv_from_svd` function in `regression.py`:

```python
def compute_efficient_pinv_from_svd(matrix, rcond_values):
    """
    Efficiently compute pseudoinverses for multiple rcond values using a single SVD.
    
    Args:
        matrix: The input matrix for which to compute pseudoinverses
        rcond_values: List of rcond values to use for thresholding
        
    Returns:
        Dictionary mapping each rcond value to its corresponding pseudoinverse matrix,
        and the singular values from the SVD
    """
    # Compute SVD once
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    
    # Get the maximum singular value for relative thresholding
    max_singular_value = S[0]
    
    # Dictionary to store results
    pinvs = {}
    
    # For each rcond value, create a custom pseudoinverse
    for rcond in rcond_values:
        # Compute threshold for this rcond value
        threshold = rcond * max_singular_value
        
        # Create reciprocal of singular values with thresholding
        S_pinv = torch.zeros_like(S)
        above_threshold = S > threshold
        S_pinv[above_threshold] = 1.0 / S[above_threshold]
        
        # Compute pseudoinverse as V * S_pinv * U.T
        S_pinv_diag = torch.diag(S_pinv)
        pinv_matrix = Vh.T @ S_pinv_diag @ U.T
        
        # Store result for this rcond value
        pinvs[rcond] = pinv_matrix
    
    return pinvs, S
```

## How to Use

### Using the Optimized Implementation

The `RegressionAnalyzer` class now supports an optional `use_efficient_pinv` parameter:

```python
# Create an analyzer with the optimized implementation
analyzer = RegressionAnalyzer(device='cpu', use_efficient_pinv=True)

# Run the analysis as usual
results, best_weights, best_singular = analyzer.analyze_checkpoint_with_activations(
    target_activations, belief_targets, rcond_sweep_list
)
```

### Comparing Both Implementations

You can compare both implementations to verify they produce equivalent results:

```python
# Run with comparison mode enabled
results, best_weights, best_singular = analyzer.analyze_checkpoint_with_activations(
    target_activations, belief_targets, rcond_sweep_list, compare_implementations=True
)
```

Alternatively, use the provided comparison script:

```bash
python -m scripts.activation_analysis.compare_pinv_methods \
    --sweep-id YOUR_SWEEP_ID \
    --run-id YOUR_RUN_ID \
    --device cuda \
    --output-csv comparison_results.csv
```

## Performance Benefits

The optimized implementation provides significant performance improvements when:
- Sweeping across multiple `rcond` values
- Processing large matrices
- Working with high-dimensional activation spaces

Expected speedup factors range from 2-5x depending on the number of `rcond` values and matrix sizes.

## Numerical Precision

The optimized implementation produces results that are numerically equivalent to the original implementation to within floating-point precision. Any differences should be extremely small (typically < 1e-12 relative difference) and not affect the final conclusions of the analysis.

## Implementation Notes

1. **Backward Compatibility**: The original implementation is preserved and remains the default behavior (`use_efficient_pinv=False`).
2. **Caching**: The implementation caches not only the SVD but also the weighted design matrices to avoid redundant computation.
3. **Fallback**: If a requested `rcond` value isn't found in the precomputed set, the implementation falls back to the original method. 