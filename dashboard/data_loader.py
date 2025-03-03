"""
Data loading utilities for the activation analysis dashboard.

This module provides functions to discover, load, and preprocess the data
from the activation analysis pipeline output for visualization.
"""

import os
import json
import glob
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up basic logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActivationAnalysisLoader:
    """
    Class to load and process activation analysis data.
    """
    
    def __init__(self, analysis_dir: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            analysis_dir: Path to the directory containing analysis outputs.
                         If None, will try several possible locations.
        """
        self.analysis_dir = self._find_analysis_dir(analysis_dir)
        logger.info(f"Using analysis directory: {self.analysis_dir}")
        self.sweeps = self._discover_sweeps()
        logger.info(f"Discovered sweeps: {self.sweeps}")
        
    def _find_analysis_dir(self, analysis_dir: Optional[str]) -> str:
        """
        Find the analysis directory by trying several possible locations.
        
        Args:
            analysis_dir: User-provided analysis directory path or None.
            
        Returns:
            Path to the analysis directory.
        """
        if analysis_dir and os.path.isdir(analysis_dir):
            return analysis_dir
            
        # Try several possible locations
        possible_locations = [
            "analysis",                          # Current directory
            "../analysis",                       # Parent directory
            os.path.join(os.getcwd(), "analysis") # Absolute path from current directory
        ]
        
        # Check if repo root directory can be detected
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)  # Assuming dashboard is directly under repo root
        possible_locations.append(os.path.join(repo_root, "analysis"))
        
        for location in possible_locations:
            if os.path.isdir(location):
                logger.info(f"Found analysis directory at: {location}")
                return location
                
        # If we get here, we couldn't find the analysis directory
        logger.warning("Could not find analysis directory in common locations. "
                     "Please specify the path explicitly.")
        
        # Default to the first option even if it doesn't exist yet
        return possible_locations[0]
        
    def _discover_sweeps(self) -> List[str]:
        """
        Discover available sweeps in the analysis directory.
        
        Returns:
            List of sweep IDs.
        """
        if not os.path.exists(self.analysis_dir):
            logger.warning(f"Analysis directory does not exist: {self.analysis_dir}")
            return []
            
        sweep_paths = [d for d in glob.glob(os.path.join(self.analysis_dir, "*")) 
                      if os.path.isdir(d) and not os.path.basename(d).endswith("logs") 
                      and not os.path.basename(d).endswith("local_cache")]
        
        sweep_ids = [os.path.basename(p) for p in sweep_paths]
        
        if not sweep_ids:
            logger.warning(f"No sweeps found in {self.analysis_dir}. "
                         f"Contents: {os.listdir(self.analysis_dir) if os.path.exists(self.analysis_dir) else 'directory does not exist'}")
        
        return sweep_ids
    
    def get_runs_in_sweep(self, sweep_id: str) -> List[str]:
        """
        Get all run IDs within a sweep.
        
        Args:
            sweep_id: ID of the sweep.
            
        Returns:
            List of run IDs.
        """
        sweep_dir = os.path.join(self.analysis_dir, sweep_id)
        
        if not os.path.exists(sweep_dir):
            logger.warning(f"Sweep directory does not exist: {sweep_dir}")
            return []
            
        run_paths = [d for d in glob.glob(os.path.join(sweep_dir, "*")) if os.path.isdir(d)]
        run_ids = [os.path.basename(p) for p in run_paths]
        
        if not run_ids:
            logger.warning(f"No runs found in sweep {sweep_id}. "
                         f"Contents: {os.listdir(sweep_dir) if os.path.exists(sweep_dir) else 'directory does not exist'}")
        
        return run_ids
    
    def load_metadata(self, sweep_id: str, run_id: str) -> Dict[str, Any]:
        """
        Load metadata for a specific run.
        
        Args:
            sweep_id: ID of the sweep.
            run_id: ID of the run.
            
        Returns:
            Dictionary containing metadata.
        """
        metadata_path = os.path.join(self.analysis_dir, sweep_id, run_id, f"{run_id}_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def load_regression_results(self, sweep_id: str, run_id: str) -> pd.DataFrame:
        """
        Load regression results for a specific run.
        
        Args:
            sweep_id: ID of the sweep.
            run_id: ID of the run.
            
        Returns:
            DataFrame containing regression results.
        """
        results_path = os.path.join(self.analysis_dir, sweep_id, run_id, f"{run_id}_regression_results.csv")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Regression results file not found: {results_path}")
        
        df = pd.read_csv(results_path)
        # Add sweep_id and run_id columns for easier filtering later
        df['sweep_id'] = sweep_id
        df['run_id'] = run_id
        
        return df
    
    def load_random_baseline(self, sweep_id: str, run_id: str) -> pd.DataFrame:
        """
        Load random baseline results for a specific run.
        
        Args:
            sweep_id: ID of the sweep.
            run_id: ID of the run.
            
        Returns:
            DataFrame containing random baseline results.
        """
        baseline_path = os.path.join(self.analysis_dir, sweep_id, run_id, f"{run_id}_random_baseline.csv")
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"Random baseline file not found: {baseline_path}")
        
        df = pd.read_csv(baseline_path)
        # Add sweep_id and run_id columns for easier filtering later
        df['sweep_id'] = sweep_id
        df['run_id'] = run_id
        
        return df
    
    def load_loss_data(self, sweep_id: str, run_id: str) -> pd.DataFrame:
        """
        Load loss data for a specific run.
        
        Args:
            sweep_id: ID of the sweep.
            run_id: ID of the run.
            
        Returns:
            DataFrame containing loss data.
        """
        loss_path = os.path.join(self.analysis_dir, sweep_id, run_id, f"{run_id}_loss.csv")
        if not os.path.exists(loss_path):
            raise FileNotFoundError(f"Loss file not found: {loss_path}")
        
        df = pd.read_csv(loss_path)
        # Add sweep_id and run_id columns for easier filtering later
        df['sweep_id'] = sweep_id
        df['run_id'] = run_id
        
        return df
    
    def parse_run_id_components(self, run_id: str) -> Dict[str, Any]:
        """
        Parse run ID to extract model configuration components.
        
        Args:
            run_id: Run identifier (e.g., "run_71_L4_H64_RNN_uni_mess3")
            
        Returns:
            Dictionary with parsed components (architecture, layers, etc.)
        """
        components = run_id.split('_')
        parsed: Dict[str, Any] = {'run_id': run_id}
        
        # Extract numerical components
        for comp in components:
            if comp.startswith('L') and comp[1:].isdigit():
                parsed['layers'] = int(comp[1:])
            elif comp.startswith('H') and comp[1:].isdigit():
                parsed['hidden_size'] = int(comp[1:])
            elif comp.startswith('DH') and comp[2:].isdigit():
                parsed['head_dim'] = int(comp[2:])
            elif comp.startswith('DM') and comp[2:].isdigit():
                parsed['model_dim'] = int(comp[2:])
        
        # Extract architecture type
        if 'RNN' in components:
            parsed['architecture'] = 'RNN'
        elif 'LSTM' in components:
            parsed['architecture'] = 'LSTM'
        elif 'GRU' in components:
            parsed['architecture'] = 'GRU'
        elif 'DH' in '_'.join(components):  # Heuristic for transformer models
            parsed['architecture'] = 'Transformer'
        
        # Extract task type
        task_identifiers = ['mess3', 'rrxor', 'fanizza', 'tom_quantum', 'post_quantum']
        for task in task_identifiers:
            if task in run_id:
                parsed['task'] = task
                break
        
        return parsed
    
    def load_all_regression_results(self, sweep_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load regression results for all runs in a sweep or all sweeps.
        
        Args:
            sweep_id: Optional sweep ID to filter by.
            
        Returns:
            Combined DataFrame with all regression results.
        """
        all_results = []
        
        if sweep_id:
            sweep_ids = [sweep_id]
        else:
            sweep_ids = self.sweeps
        
        for sid in sweep_ids:
            runs = self.get_runs_in_sweep(sid)
            for run_id in runs:
                try:
                    df = self.load_regression_results(sid, run_id)
                    # Add parsed components
                    parsed_components = self.parse_run_id_components(run_id)
                    for key, value in parsed_components.items():
                        if key != 'run_id':  # Already added
                            df[key] = value
                    all_results.append(df)
                except FileNotFoundError:
                    print(f"Warning: Could not load regression results for {sid}/{run_id}")
        
        if not all_results:
            return pd.DataFrame()
        
        return pd.concat(all_results, ignore_index=True)
    
    def get_model_comparison_data(self, sweep_id: Optional[str] = None, 
                                 metric: str = 'r_squared', 
                                 layer_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Get data formatted for comparing models across architectures and tasks.
        
        Args:
            sweep_id: Optional sweep ID to filter by.
            metric: Metric to compare (default: 'r_squared').
            layer_idx: Optional layer index to filter by.
            
        Returns:
            DataFrame formatted for model comparison visualization.
        """
        df = self.load_all_regression_results(sweep_id)
        
        if df.empty:
            return df
        
        # Filter by layer if specified
        if layer_idx is not None:
            df = df[df['layer_idx'] == layer_idx]
        
        # Group by relevant dimensions and calculate means
        grouped = df.groupby(['architecture', 'task', 'checkpoint'])[metric].mean().reset_index()
        
        return grouped
    
    def get_best_rcond_data(self, regression_df):
        """
        For each checkpoint, target, and layer, select the data with the best rcond value
        (the one that gives the lowest norm_dist).
        
        Args:
            regression_df: DataFrame containing regression results
            
        Returns:
            DataFrame with only the best rcond data rows
        """
        # Group by the relevant dimensions
        # For each group, take the row with the minimum norm_dist value
        grouped = regression_df.groupby(['checkpoint', 'target', 'layer_idx', 'layer_name'])
        
        # Initialize list to collect optimal rows
        best_rows = []
        
        # For each group, find the row with the minimum norm_dist
        for (checkpoint, target, layer_idx, layer_name), group in grouped:
            best_row = group.loc[group['norm_dist'].idxmin()]
            best_rows.append(best_row)
        
        # Create a new DataFrame with the best rows
        best_df = pd.DataFrame(best_rows)
        
        return best_df
    
    def normalize_by_random_baseline(self, regression_df, random_df):
        """
        Normalize norm_dist values by dividing by the mean of random networks,
        doing this separately for each target and layer.
        
        Args:
            regression_df: DataFrame containing regression results
            random_df: DataFrame containing random baseline results
            
        Returns:
            DataFrame with normalized norm_dist values
        """
        # Process random data to get the best rcond rows
        random_best = self.get_best_rcond_data(random_df)
        
        # Calculate mean norm_dist for each target and layer in random data
        random_means = random_best.groupby(['target', 'layer_idx'])['norm_dist'].mean().reset_index()
        random_means = random_means.rename(columns={'norm_dist': 'random_mean_norm_dist'})
        
        # Merge the random means with the regression data
        normalized_df = pd.merge(
            regression_df,
            random_means,
            on=['target', 'layer_idx'],
            how='left'
        )
        
        # Calculate normalized norm_dist
        normalized_df['norm_dist_normalized'] = normalized_df['norm_dist'] / normalized_df['random_mean_norm_dist']
        
        return normalized_df
    
    def load_processed_regression_results(self, sweep_id, run_id):
        """
        Load regression results and process them to:
        1. Select the best rcond data for each checkpoint, target, and layer
        2. Normalize norm_dist by random baseline means
        
        Args:
            sweep_id: ID of the sweep
            run_id: ID of the run
            
        Returns:
            DataFrame with processed regression results
        """
        # Load raw regression results
        regression_df = self.load_regression_results(sweep_id, run_id)
        
        # Select data with best rcond values
        best_df = self.get_best_rcond_data(regression_df)
        
        try:
            # Load random baseline and normalize
            random_df = self.load_random_baseline(sweep_id, run_id)
            normalized_df = self.normalize_by_random_baseline(best_df, random_df)
            return normalized_df
            
        except FileNotFoundError:
            logger.warning(f"Random baseline data not found for {sweep_id}/{run_id}. "
                         f"Returning non-normalized data.")
            return best_df


# Example usage
if __name__ == "__main__":
    loader = ActivationAnalysisLoader()
    print(f"Available sweeps: {loader.sweeps}")
    
    if loader.sweeps:
        sweep_id = loader.sweeps[0]
        runs = loader.get_runs_in_sweep(sweep_id)
        print(f"Runs in sweep {sweep_id}: {runs}")
        
        if runs:
            run_id = runs[0]
            metadata = loader.load_metadata(sweep_id, run_id)
            print(f"Metadata for {run_id}: {metadata}")
            
            comparison_data = loader.get_model_comparison_data(sweep_id)
            print(f"Model comparison data shape: {comparison_data.shape}") 