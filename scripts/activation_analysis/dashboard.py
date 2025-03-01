import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import h5py
import os
import glob
import json
from io import StringIO  # Add import for StringIO
import traceback
import logging

# Define metrics with their display information
METRICS = {
    'norm_dist': {
        'title': 'Distance',
        'description': 'Lower is better'
    },
    'r_squared': {
        'title': 'R²',
        'description': 'Higher is better'
    },
    'variance_explained': {
        'title': 'Var. Explained',
        'description': 'Higher is better'
    },
    'dims': {
        'title': 'Dimensions',
        'description': 'Number of components'
    }
}

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "Activation Analysis Dashboard"

# Set up the layout
app.layout = html.Div([
    html.Div([
        html.H1("Activation Analysis Dashboard", style={'textAlign': 'center'}),
        html.Hr(),
    ], className="row"),
    
    html.Div([
        html.Div([
            html.H4("Select Data"),
            html.Label("Data Directory:"),
            dcc.Input(
                id="data-directory",
                type="text",
                value="./analysis",  # Default directory from config
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            html.Button('Load Runs', id='load-runs-button', n_clicks=0),
            
            html.Hr(),
            
            html.Label("Run ID:"),
            dcc.Dropdown(id='run-id-dropdown'),
            
            html.Label("Target:"),
            dcc.Dropdown(id='target-dropdown'),
            
            html.Label("Checkpoint:"),
            dcc.Dropdown(id='checkpoint-dropdown', multi=True),
            
            # Changed from slider to dropdown for more precise selection
            html.Label("Regularization Parameter (rcond):"),
            dcc.Dropdown(
                id='rcond-dropdown',
                options=[],
                value=None
            ),
            
            html.Hr(),
            
            html.Label("Layer:"),
            dcc.Dropdown(id='layer-dropdown'),
            
            # Add multi-layer selection dropdown for checkpoint evolution plot
            html.Label("Multiple Layers (for Training Dynamics):"), 
            dcc.Dropdown(
                id='multi-layer-dropdown',
                multi=True
            ),
            
            # Global metric selector
            html.Hr(),
            html.Label("Display Metric:"),
            dcc.RadioItems(
                id='metric-selector',
                options=[
                    {'label': 'Normalized Distance (lower is better)', 'value': 'norm_dist'},
                    {'label': 'R² (higher is better)', 'value': 'r_squared'}
                ],
                value='norm_dist',  # Default to normalized distance
                labelStyle={'display': 'block', 'margin-bottom': '5px'}
            ),
            
            # Distance normalization for random baseline
            html.Label("Distance Normalization (Random Baseline):"),
            dcc.RadioItems(
                id='distance-normalization',
                options=[
                    {'label': 'Raw Distances', 'value': 'raw'},
                    {'label': 'Normalized by Random', 'value': 'normalized'}
                ],
                value='raw',
                labelStyle={'display': 'block', 'margin-bottom': '5px'}
            ),
            
            html.Hr(),
            
            html.Button('Update Plots', id='update-button', n_clicks=0),
            
            html.Hr(),
            
            html.Div(id='dataset-info')
        ], className="three columns", style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        html.Div([
            # Tabs for different visualization sections
            dcc.Tabs([
                dcc.Tab(label="Performance Overview", children=[
                    html.Div([
                        html.H4("Layer Performance Comparison", style={'textAlign': 'center'}),
                        dcc.Graph(id='layer-performance-chart'),
                    ], className="row"),
                    
                    html.Div([
                        html.Div([
                            html.H4("Regularization Effect", style={'textAlign': 'center'}),
                            dcc.Graph(id='regularization-curve'),
                        ], className="six columns"),
                        
                        html.Div([
                            html.H4("Layer vs Target Heatmap", style={'textAlign': 'center'}),
                            dcc.Graph(id='layer-target-heatmap'),
                            html.Div([
                                html.Label("Regularization Parameter (rcond):"),
                                dcc.Input(id='rcond-input', type='text', value='1e-3'),
                                html.Label("Metric:"),
                                dcc.Dropdown(
                                    id='metric-dropdown',
                                    options=[
                                        {'label': 'Normalized Distance', 'value': 'norm_dist'},
                                        {'label': 'R²', 'value': 'r_squared'},
                                        {'label': 'MSE', 'value': 'mse'}
                                    ],
                                    value='norm_dist'
                                ),
                                html.Label("Normalization:"),
                                dcc.Dropdown(
                                    id='normalization-dropdown',
                                    options=[
                                        {'label': 'None', 'value': 'none'},
                                        {'label': 'Z-Score', 'value': 'z-score'},
                                        {'label': 'Min-Max', 'value': 'min-max'}
                                    ],
                                    value='none'
                                ),
                                html.Button('Update Heatmap', id='update-plots-button', n_clicks=0, style={'margin-top': '10px'})
                            ], style={'margin-top': '10px'}),
                        ], className="six columns"),
                    ], className="row"),
                ]),
                
                dcc.Tab(label="Training Dynamics", children=[
                    html.Div([
                        html.H4("Checkpoint Evolution", style={'textAlign': 'center'}),
                        dcc.Graph(id='checkpoint-evolution'),
                    ], className="row"),
                    
                    html.Div([
                        html.H4("R² vs Distance Metrics", style={'textAlign': 'center'}),
                        dcc.Graph(id='r-squared-vs-dist'),
                    ], className="row"),
                ]),
                
                dcc.Tab(label="SVD Analysis", children=[
                    html.Div([
                        html.H4("Singular Value Distribution", style={'textAlign': 'center'}),
                        dcc.Graph(id='singular-value-plot'),
                    ], className="row"),
                    
                    html.Div([
                        html.H4("Cumulative Variance Explained", style={'textAlign': 'center'}),
                        dcc.Graph(id='cumulative-variance'),
                    ], className="row"),
                ]),
                
                dcc.Tab(label="Random Baseline", children=[
                    html.Div([
                        html.H4("Comparison to Random Baseline", style={'textAlign': 'center'}),
                        dcc.Graph(id='random-baseline-comparison'),
                    ], className="row"),
                ]),
                
                dcc.Tab(label="Debug View", children=[
                    html.Div([
                        html.H4("Debug Information", style={'textAlign': 'center'}),
                        html.Button('Run Diagnostics', id='run-diagnostics-button', n_clicks=0),
                        html.Div(id='debug-output', style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'backgroundColor': '#f5f5f5', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px', 'overflow': 'auto', 'maxHeight': '600px'}),
                    ], className="row"),
                ]),
            ]),
        ], className="nine columns"),
    ], className="row"),
    
    # Store components for intermediate data
    dcc.Store(id='runs-data'),
    dcc.Store(id='current-df'),
    dcc.Store(id='singular-values-data'),
    dcc.Store(id='weights-data'),
    dcc.Store(id='rcond-options'),
    dcc.Store(id='random-baseline-data'),
    
    # Debug info div - show hidden by default, unhide for debugging
    html.Div(id='debug-info', style={'display': 'none'}),
])

# Helper functions for file operations
def find_csv_files(directory, include_random=True):
    """Find all CSV result files in the specified directory and its subdirectories."""
    result_files = []
    random_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                if '_random_baseline' in file and not file.endswith('_weights.csv') and not file.endswith('_singular_values.csv'):
                    random_files.append(os.path.join(root, file))
                elif file.endswith('_results.csv'):
                    result_files.append(os.path.join(root, file))
    
    if include_random:
        return result_files, random_files
    else:
        return result_files

def extract_run_id(filename):
    """Extract run_id from the filename."""
    base = os.path.basename(filename)
    if '_random_baseline' in base:
        run_id = base.split('_random_baseline')[0]
    else:
        run_id = base.split('_results')[0]
    return run_id

def find_related_files(base_file):
    """Find related CSV files for a given base results file."""
    directory = os.path.dirname(base_file)
    base_name = os.path.basename(base_file)
    run_id = extract_run_id(base_name)
    
    # Find weights and singular values files
    weights_file = os.path.join(directory, f"{run_id}_weights.csv")
    singular_file = os.path.join(directory, f"{run_id}_singular_values.csv")
    metadata_file = os.path.join(directory, f"{run_id}_metadata.json")
    
    # Find random baseline if it exists
    random_file = os.path.join(directory, f"{run_id}_random_baseline.csv")
    if not os.path.exists(random_file):
        random_file = None
    
    return {
        'results': base_file,
        'weights': weights_file if os.path.exists(weights_file) else None,
        'singular_values': singular_file if os.path.exists(singular_file) else None,
        'metadata': metadata_file if os.path.exists(metadata_file) else None,
        'random': random_file
    }

def load_csv_dataframe(csv_file):
    """Load regression results from CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(csv_file)
        
        # Convert string columns to proper types if needed
        if 'random_baseline' in df.columns and df['random_baseline'].dtype == object:
            df['random_baseline'] = df['random_baseline'].apply(
                lambda x: x.lower() == 'true' if isinstance(x, str) else bool(x)
            )
        
        return df
    except Exception as e:
        print(f"Error loading CSV file {csv_file}: {e}")
        return None

def load_weights_dataframe(weights_file):
    """
    Load weights data from a CSV file.
    
    Args:
        weights_file: Path to the weights CSV file
    
    Returns:
        dict: Dictionary of layer -> checkpoint -> weights data
    """
    try:
        # Load the CSV file into a DataFrame
        weights_df = pd.read_csv(weights_file)
        
        # Initialize result dictionary
        weights_dict = {}
        
        # Process each row in the DataFrame
        for _, row in weights_df.iterrows():
            layer = row['layer_name']
            checkpoint = row['checkpoint']
            
            # Parse weights from JSON string
            weights_data = json.loads(row['weights_data'])
            
            # Convert to numpy array
            weights_array = np.array(weights_data)
            
            # Create data dictionary
            data = {
                'weights': weights_array,
                'dims': row.get('dims', str(weights_array.shape)),
                'checkpoint': checkpoint
            }
            
            # Add optional fields if present
            for field in ['rcond', 'dist', 'random_idx']:
                if field in row and not pd.isna(row[field]):
                    data[field] = row[field]
            
            # Add is_random flag
            data['is_random'] = bool(row.get('is_random', False))
            
            # Add to weights dictionary
            if layer not in weights_dict:
                weights_dict[layer] = {}
            
            weights_dict[layer][checkpoint] = data
        
        return weights_dict
        
    except Exception as e:
        logging.error(f"Error loading weights from {weights_file}: {e}")
        logging.error(traceback.format_exc())
        return {}

def normalize_data(df, normalization_type):
    """
    Normalize data in the dataframe based on the specified normalization type.
    
    Args:
        df: DataFrame containing the data
        normalization_type: Type of normalization ('z-score', 'min-max', 'none')
        
    Returns:
        Normalized DataFrame
    """
    if normalization_type == 'none' or df.empty:
        return df
    
    # Create a copy of the dataframe to avoid modifying the original
    normalized_df = df.copy()
    
    # Determine which columns to normalize
    metrics = ['norm_dist', 'r_squared', 'variance_explained']
    normalize_cols = [col for col in metrics if col in df.columns]
    
    if not normalize_cols:
        return normalized_df
    
    # Apply normalization by group (target + layer)
    group_cols = ['target', 'layer_name']
    
    # Check if group columns exist
    existing_group_cols = [col for col in group_cols if col in df.columns]
    if not existing_group_cols:
        return normalized_df
    
    # Group by target and layer, then normalize
    for names, group_df in normalized_df.groupby(existing_group_cols):
        group_indices = group_df.index
        
        for col in normalize_cols:
            values = group_df[col].values
            
            if normalization_type == 'z-score':
                # Z-score normalization
                mean = np.mean(values)
                std = np.std(values)
                if std > 0:  # Avoid division by zero
                    normalized_df.loc[group_indices, col] = (values - mean) / std
            
            elif normalization_type == 'min-max':
                # Min-max normalization
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val > min_val:  # Avoid division by zero
                    normalized_df.loc[group_indices, col] = (values - min_val) / (max_val - min_val)
    
    return normalized_df

def load_singular_values(singular_file):
    """Load singular values data from CSV file."""
    if not singular_file or not os.path.exists(singular_file):
        return {}
    
    try:
        df = pd.read_csv(singular_file)
        sv_dict = {}
        
        for _, row in df.iterrows():
            target = row['target']
            layer = row['layer']
            sv_data = json.loads(row['singular_data'])
            
            if target not in sv_dict:
                sv_dict[target] = {}
            
            if layer not in sv_dict[target]:
                sv_dict[target][layer] = []
            
            entry = {
                'singular_values': np.array(sv_data['singular_values'])
            }
            
            if 'checkpoint' in sv_data:
                entry['checkpoint'] = sv_data['checkpoint']
            if 'random_idx' in sv_data:
                entry['random_idx'] = sv_data['random_idx']
            
            sv_dict[target][layer].append(entry)
        
        return sv_dict
    except Exception as e:
        print(f"Error loading singular values file {singular_file}: {e}")
        return {}

# Keep the h5 functions for backward compatibility
def find_h5_files(directory, include_random=True):
    """Find all h5 files in the specified directory and its subdirectories."""
    h5_files = []
    random_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                if '_random_baseline' in file:
                    random_files.append(os.path.join(root, file))
                else:
                    h5_files.append(os.path.join(root, file))
    
    if include_random:
        return h5_files, random_files
    else:
        return h5_files

def load_h5_dataframe(h5_file):
    """Load regression results from H5 file and convert to a DataFrame."""
    try:
        with h5py.File(h5_file, 'r') as f:
            if 'regression_results' not in f:
                return None
            
            # Extract data from H5 file
            df_dict = {}
            for col in f['regression_results']:
                data = f['regression_results'][col][()]
                # Convert bytes to strings if necessary
                if data.dtype.kind == 'S':
                    data = np.array([s.decode('utf-8') for s in data])
                df_dict[col] = data
            
            df = pd.DataFrame(df_dict)
            return df
    except Exception as e:
        print(f"Error loading H5 file {h5_file}: {e}")
        return None

def load_unified_h5_data(h5_file):
    """
    Load data from a unified H5 file format.
    
    Parameters:
    ----------
    h5_file : str
        Path to the unified H5 file
        
    Returns:
    -------
    dict
        Dictionary containing the extracted data
    """
    try:
        result = {
            'checkpoint_df': None,
            'random_df': None,
            'weights': None,
            'singular_values': None,
            'random_singular_values': None
        }
        
        with h5py.File(h5_file, 'r') as f:
            # Check if this is a unified format file
            if 'unified_format' not in f.attrs or not f.attrs['unified_format']:
                logger.warning(f"File {h5_file} is not in unified format")
                return None
            
            # Get global attributes to add back to dataframes
            run_id = f.attrs.get('run_id', None)
            sweep_id = f.attrs.get('sweep_id', None)
            
            # Load checkpoint results
            if 'checkpoint_results' in f:
                checkpoint_group = f['checkpoint_results']
                
                # Load regression results
                if 'regression_results' in checkpoint_group:
                    df_dict = {}
                    for col in checkpoint_group['regression_results']:
                        data = checkpoint_group['regression_results'][col][()]
                        if data.dtype.kind == 'S':
                            data = np.array([s.decode('utf-8') for s in data])
                        df_dict[col] = data
                    
                    # Create dataframe and add global attributes back if they were in the file
                    df = pd.DataFrame(df_dict)
                    if run_id is not None and 'run_id' not in df.columns:
                        df['run_id'] = run_id
                    if sweep_id is not None and 'sweep_id' not in df.columns:
                        df['sweep_id'] = sweep_id
                    
                    # Set flag to identify checkpoint data
                    df['is_random'] = False
                    
                    result['checkpoint_df'] = df
                
                # Load weights (with new checkpoint-first structure)
                if 'best_weights' in checkpoint_group:
                    weights_dict = {}
                    
                    for target in checkpoint_group['best_weights']:
                        target_group = checkpoint_group['best_weights'][target]
                        weights_dict[target] = {}
                        
                        # Iterate through each checkpoint group
                        for checkpoint in target_group:
                            checkpoint_group = target_group[checkpoint]
                            
                            # Iterate through each layer in this checkpoint
                            for layer in checkpoint_group:
                                layer_group = checkpoint_group[layer]
                                
                                # Create a unique layer key for this checkpoint-layer combination
                                layer_key = f"{layer}_cp{checkpoint}" if checkpoint != "best" else layer
                                
                                weights_dict[target][layer_key] = {
                                    "weights": layer_group['weights'][()],
                                    "checkpoint": int(checkpoint) if checkpoint.isdigit() else checkpoint,
                                    "rcond": layer_group.attrs['rcond'],
                                    "dist": layer_group.attrs['dist']
                                }
                    
                    result['weights'] = weights_dict
                
                # Load singular values
                if 'singular_values' in checkpoint_group:
                    sv_dict = {}
                    
                    for target in checkpoint_group['singular_values']:
                        target_group = checkpoint_group['singular_values'][target]
                        sv_dict[target] = {}
                        
                        for layer in target_group:
                            layer_group = target_group[layer]
                            sv_dict[target][layer] = []
                            
                            for entry_name in layer_group:
                                entry = layer_group[entry_name]
                                
                                # Extract checkpoint number from group name if possible
                                checkpoint_id = None
                                if entry_name.startswith('checkpoint_'):
                                    try:
                                        checkpoint_id = entry_name.split('_')[1]
                                    except:
                                        pass
                                
                                entry_data = {
                                    "singular_values": entry['singular_values'][()]
                                }
                                
                                # Use extracted checkpoint ID or get from attributes
                                if checkpoint_id is not None:
                                    entry_data['checkpoint'] = checkpoint_id
                                elif "checkpoint" in entry.attrs:
                                    entry_data['checkpoint'] = entry.attrs['checkpoint']
                                
                                sv_dict[target][layer].append(entry_data)
                    
                    result['singular_values'] = sv_dict
            
            # Load random baseline results
            if 'random_results' in f:
                random_group = f['random_results']
                
                # Load regression results
                if 'regression_results' in random_group:
                    df_dict = {}
                    for col in random_group['regression_results']:
                        data = random_group['regression_results'][col][()]
                        if data.dtype.kind == 'S':
                            data = np.array([s.decode('utf-8') for s in data])
                        df_dict[col] = data
                    
                    # Create dataframe and add global attributes back if they were in the file
                    df = pd.DataFrame(df_dict)
                    if run_id is not None and 'run_id' not in df.columns:
                        df['run_id'] = run_id
                    if sweep_id is not None and 'sweep_id' not in df.columns:
                        df['sweep_id'] = sweep_id
                    
                    # Set flag to identify random data
                    df['is_random'] = True
                    
                    result['random_df'] = df
                
                # Load singular values
                if 'singular_values' in random_group:
                    sv_dict = {}
                    
                    for target in random_group['singular_values']:
                        target_group = random_group['singular_values'][target]
                        sv_dict[target] = {}
                        
                        for layer in target_group:
                            layer_group = target_group[layer]
                            sv_dict[target][layer] = []
                            
                            for entry_name in layer_group:
                                entry = layer_group[entry_name]
                                
                                # Extract random index from group name if possible
                                random_id = None
                                if entry_name.startswith('random_'):
                                    try:
                                        random_id = int(entry_name.split('_')[1])
                                    except:
                                        pass
                                
                                entry_data = {
                                    "singular_values": entry['singular_values'][()]
                                }
                                
                                # Use extracted random ID or get from attributes
                                if random_id is not None:
                                    entry_data['random_idx'] = random_id
                                elif "random_idx" in entry.attrs:
                                    entry_data['random_idx'] = entry.attrs['random_idx']
                                
                                sv_dict[target][layer].append(entry_data)
                    
                    result['random_singular_values'] = sv_dict
        
        return result
    except Exception as e:
        logger.error(f"Error loading unified H5 data: {e}")
        return None

def get_singular_values(h5_file):
    """Extract singular values from H5 file."""
    singular_values = {}
    with h5py.File(h5_file, 'r') as f:
        # Check if unified format first
        if 'unified_format' in f.attrs and f.attrs['unified_format'] is True:
            # Handle unified format
            if 'checkpoint_results' in f and 'singular_values' in f['checkpoint_results']:
                sv_group = f['checkpoint_results/singular_values']
                
                for target in sv_group.keys():
                    target_group = sv_group[target]
                    singular_values[target] = {}
                    
                    for layer in target_group.keys():
                        layer_group = target_group[layer]
                        singular_values[target][layer] = []
                        
                        for entry_key in layer_group.keys():
                            entry = layer_group[entry_key]
                            
                            # Extract checkpoint number from group name if possible
                            checkpoint_id = None
                            if entry_key.startswith('checkpoint_'):
                                try:
                                    checkpoint_id = entry_key.split('_')[1]
                                except:
                                    pass
                            
                            entry_data = {
                                'singular_values': entry['singular_values'][:].tolist()
                            }
                            
                            # Use extracted checkpoint ID or get from attributes
                            if checkpoint_id is not None:
                                entry_data['checkpoint'] = checkpoint_id
                            elif 'checkpoint' in entry.attrs:
                                entry_data['checkpoint'] = str(entry.attrs['checkpoint'])
                                
                            singular_values[target][layer].append(entry_data)
            
            # Also check for random singular values
            if 'random_results' in f and 'singular_values' in f['random_results']:
                sv_group = f['random_results/singular_values']
                
                for target in sv_group.keys():
                    if target not in singular_values:
                        singular_values[target] = {}
                    
                    for layer in sv_group[target].keys():
                        if layer not in singular_values[target]:
                            singular_values[target][layer] = []
                        
                        for entry_key in sv_group[target][layer].keys():
                            entry = sv_group[target][layer][entry_key]
                            
                            # Extract random index from group name if possible
                            random_id = None
                            if entry_key.startswith('random_'):
                                try:
                                    random_id = int(entry_key.split('_')[1])
                                except:
                                    pass
                            
                            entry_data = {
                                'singular_values': entry['singular_values'][:].tolist()
                            }
                            
                            # Use extracted random ID or get from attributes
                            if random_id is not None:
                                entry_data['random_idx'] = random_id
                            elif 'random_idx' in entry.attrs:
                                entry_data['random_idx'] = entry.attrs['random_idx']
                                
                            singular_values[target][layer].append(entry_data)
        else:
            # Legacy format - handle the old way
        if 'singular_values' not in f:
            return singular_values
        
        sv_group = f['singular_values']
        
        for target in sv_group.keys():
            target_group = sv_group[target]
            singular_values[target] = {}
            
            for layer in target_group.keys():
                layer_group = target_group[layer]
                singular_values[target][layer] = []
                
                for entry_key in layer_group.keys():
                    entry = layer_group[entry_key]
                    entry_data = {
                        'singular_values': entry['singular_values'][:].tolist()
                    }
                    
                    if 'checkpoint' in entry.attrs:
                        entry_data['checkpoint'] = str(entry.attrs['checkpoint'])
                    if 'random_idx' in entry.attrs:
                        entry_data['random_idx'] = entry.attrs['random_idx']
                    
                    singular_values[target][layer].append(entry_data)
    
    return singular_values

def get_best_weights(h5_file):
    """Extract best weights from H5 file."""
    best_weights = {}
    with h5py.File(h5_file, 'r') as f:
        if 'best_weights' not in f:
            return best_weights
        
        weights_group = f['best_weights']
        
        for target in weights_group.keys():
            target_group = weights_group[target]
            best_weights[target] = {}
            
            for layer in target_group.keys():
                layer_group = target_group[layer]
                best_weights[target][layer] = {
                    'weights': layer_group['weights'][:].tolist(),
                    'checkpoint': str(layer_group.attrs.get('checkpoint', 'unknown')),
                    'rcond': float(layer_group.attrs.get('rcond', 0)),
                    'dist': float(layer_group.attrs.get('dist', 0))
                }
    
    return best_weights

def find_best_rcond(df, target, checkpoint, layer, metric='norm_dist', better='min'):
    """Find the best rcond value for a given target/checkpoint/layer combination."""
    # Filter dataframe for the given target/checkpoint/layer
    filtered_df = df[
        (df['target'] == target) &
        (df['checkpoint'] == str(checkpoint)) &
        (df['layer_name'] == layer)
    ]
    
    if filtered_df.empty:
        return None
    
    # Find the best rcond based on the metric
    if better == 'min':  # For metrics where lower is better (e.g., norm_dist)
        idx = filtered_df[metric].idxmin()
    else:  # For metrics where higher is better (e.g., r_squared)
        idx = filtered_df[metric].idxmax()
    
    best_rcond = filtered_df.loc[idx, 'rcond']
    
    return best_rcond

def find_best_rconds_for_all_layers(df, target, checkpoint, layers, metric='norm_dist', better='min'):
    """Find the best rcond value for each layer."""
    best_rconds = {}
    
    for layer in layers:
        best_rcond = find_best_rcond(df, target, checkpoint, layer, metric, better)
        if best_rcond is not None:
            best_rconds[layer] = best_rcond
    
    return best_rconds

def get_metric_info(metric):
    """Return information about a metric."""
    if metric == 'norm_dist':
        return {
            'title': 'Normalized Distance',
            'axis_title': 'Normalized Distance',
            'better': 'min',
            'description': 'lower is better'
        }
    elif metric == 'r_squared':
        return {
            'title': 'R²',
            'axis_title': 'R²',
            'better': 'max',
            'description': 'higher is better'
        }
    else:
        return {
            'title': metric,
            'axis_title': metric,
            'better': 'unknown',
            'description': ''
        }

# Helper function to convert JSON string to DataFrame using StringIO
def json_to_dataframe(df_json):
    """Convert JSON string to dataframe using StringIO to avoid FutureWarning."""
    return pd.read_json(StringIO(df_json), orient='split')

def find_unified_h5_files(directory):
    """Find unified H5 files in the directory."""
    unified_h5_files = []
    pattern = os.path.join(directory, '*_unified_results.h5')
    for file_path in glob.glob(pattern):
        # Verify this is actually a unified format file
        try:
            with h5py.File(file_path, 'r') as f:
                if 'unified_format' in f.attrs and f.attrs['unified_format']:
                    unified_h5_files.append(file_path)
        except Exception as e:
            print(f"Error checking unified file {file_path}: {e}")
    return unified_h5_files

def find_csv_npy_files(directory):
    """Find CSV+NPY format files in the directory."""
    csv_files = {}
    random_csv_files = {}
    metadata_files = []
    
    # Look for metadata JSON files
    pattern = os.path.join(directory, '*_metadata.json')
    for file_path in glob.glob(pattern):
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
                
            # Check if this is a CSV+NPY format file
            if metadata.get('format_version', '').startswith('csv_npy_'):
                run_id = metadata.get('run_id')
                if run_id:
                    metadata_files.append(file_path)
                    
                    # Get the results CSV path
                    results_csv = metadata.get('file_index', {}).get('results_csv')
                    if results_csv and os.path.exists(results_csv):
                        csv_files[run_id] = {
                            'type': 'csv_npy',
                            'metadata': file_path,
                            'results': results_csv
                        }
                    
                    # Get the random results CSV path
                    random_csv = metadata.get('file_index', {}).get('random_results_csv')
                    if random_csv and os.path.exists(random_csv):
                        random_csv_files[run_id] = {
                            'type': 'csv_npy',
                            'metadata': file_path,
                            'results': random_csv
                        }
        except Exception as e:
            print(f"Error checking CSV+NPY file {file_path}: {e}")
    
    # If no metadata files were found, look for standalone CSV files
    if not metadata_files:
        csv_pattern = os.path.join(directory, '*_results.csv')
        for file_path in glob.glob(csv_pattern):
            if not '_random_results.csv' in file_path:
                run_id = os.path.basename(file_path).replace('_results.csv', '')
                csv_files[run_id] = {
                    'type': 'csv',
                    'results': file_path
                }
        
        random_csv_pattern = os.path.join(directory, '*_random_results.csv')
        for file_path in glob.glob(random_csv_pattern):
            run_id = os.path.basename(file_path).replace('_random_results.csv', '')
            random_csv_files[run_id] = {
                'type': 'csv',
                'results': file_path
            }
    
    return csv_files, random_csv_files

def load_csv_npy_data(run_id, base_dir):
    """
    Load data from the CSV+NPY format.
    
    Parameters:
    ----------
    run_id : str
        Run identifier
    base_dir : str
        Base directory where the data is stored
        
    Returns:
    -------
    dict
        Dictionary containing the loaded data
    """
    try:
        # Initialize the result structure
        result = {
            'checkpoint_df': None,
            'random_df': None,
            'weights': {},
            'singular_values': {},
            'random_singular_values': {}
        }
        
        # Load metadata
        metadata_path = os.path.join(base_dir, f"{run_id}_metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: {metadata_path}")
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load checkpoint results CSV
        results_csv = metadata.get('file_index', {}).get('results_csv')
        if results_csv and os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            # Ensure is_random flag is set
            if 'is_random' not in df.columns:
                df['is_random'] = False
            result['checkpoint_df'] = df
        
        # Load random results CSV
        random_csv = metadata.get('file_index', {}).get('random_results_csv')
        if random_csv and os.path.exists(random_csv):
            df = pd.read_csv(random_csv)
            # Ensure is_random flag is set
            if 'is_random' not in df.columns:
                df['is_random'] = True
            result['random_df'] = df
        
        # Load weights
        weights_data = metadata.get('file_index', {}).get('weights', {})
        for target, target_data in weights_data.items():
            result['weights'][target] = {}
            for layer, layer_data in target_data.items():
                result['weights'][target][layer] = {}
                for checkpoint_key, file_info in layer_data.items():
                    is_random = file_info.get('is_random', False)
                    weight_path = file_info.get('path')
                    
                    if not weight_path or not os.path.exists(weight_path):
                        continue
                        
                    try:
                        weights = np.load(weight_path)
                        entry = {
                            'weights': weights
                        }
                        
                        # Add metadata from file_info
                        for key, value in file_info.items():
                            if key not in ['path', 'shape']:
                                entry[key] = value
                        
                        # Parse checkpoint or random_idx
                        if is_random:
                            random_idx = file_info.get('random_idx')
                            if random_idx is not None:
                                entry['random_idx'] = random_idx
                        else:
                            # If checkpoint_key is a number, convert it to int
                            try:
                                checkpoint = int(checkpoint_key)
                                entry['checkpoint'] = checkpoint
                            except ValueError:
                                pass
                        
                        # Store in appropriate structure based on random/checkpoint
                        if is_random:
                            # For random weights, add to a list
                            if target not in result['random_weights']:
                                result['random_weights'][target] = {}
                            if layer not in result['random_weights'][target]:
                                result['random_weights'][target][layer] = []
                            result['random_weights'][target][layer].append(entry)
                        else:
                            # For checkpoint weights, store directly
                            result['weights'][target][layer] = entry
                    except Exception as e:
                        logger.error(f"Error loading weights from {weight_path}: {e}")
        
        # Load singular values
        sv_data = metadata.get('file_index', {}).get('singular_values', {})
        for target, target_data in sv_data.items():
            result['singular_values'][target] = {}
            result['random_singular_values'][target] = {}
            
            for layer, layer_data in target_data.items():
                result['singular_values'][target][layer] = []
                result['random_singular_values'][target][layer] = []
                
                for checkpoint_key, file_info in layer_data.items():
                    is_random = file_info.get('is_random', False)
                    sv_path = file_info.get('path')
                    
                    if not sv_path or not os.path.exists(sv_path):
                        continue
                        
                    try:
                        sv_array = np.load(sv_path)
                        entry = {
                            'singular_values': sv_array
                        }
                        
                        # Add metadata from file_info
                        for key, value in file_info.items():
                            if key not in ['path', 'shape']:
                                entry[key] = value
                        
                        # Parse checkpoint or random_idx
                        if is_random:
                            random_idx = file_info.get('random_idx')
                            if random_idx is not None:
                                entry['random_idx'] = random_idx
                        else:
                            # If checkpoint_key is a number, convert it to int
                            try:
                                checkpoint = int(checkpoint_key)
                                entry['checkpoint'] = checkpoint
                            except ValueError:
                                pass
                        
                        # Store in appropriate structure based on random/checkpoint
                        if is_random:
                            result['random_singular_values'][target][layer].append(entry)
                        else:
                            result['singular_values'][target][layer].append(entry)
                    except Exception as e:
                        logger.error(f"Error loading singular values from {sv_path}: {e}")
        
        return result
    except Exception as e:
        logger.error(f"Error loading CSV+NPY data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Update the find_unified_h5_files function to also look for CSV+NPY files
def find_unified_h5_files(directory):
    """Find unified H5 files or CSV+NPY files in the directory."""
    files = []
    
    # First look for H5 files
    h5_pattern = os.path.join(directory, '*_unified_results.h5')
    for file_path in glob.glob(h5_pattern):
        try:
            with h5py.File(file_path, 'r') as f:
                if 'unified_format' in f.attrs and f.attrs['unified_format']:
                    files.append(file_path)
        except Exception as e:
            print(f"Error checking unified H5 file {file_path}: {e}")
    
    # Then look for metadata files from CSV+NPY format
    json_pattern = os.path.join(directory, '*_metadata.json')
    for file_path in glob.glob(json_pattern):
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            if 'format_version' in metadata and metadata['format_version'].startswith('csv_npy_'):
                files.append(file_path)
        except Exception as e:
            print(f"Error checking metadata file {file_path}: {e}")
    
    return files

# Callback to load available runs
@app.callback(
    [Output('runs-data', 'data'),
     Output('run-id-dropdown', 'options'),
     Output('dataset-info', 'children'),
     Output('random-baseline-data', 'data')],
    [Input('load-runs-button', 'n_clicks')],
    [State('data-directory', 'value')]
)
def load_available_runs(n_clicks, directory):
    """Load available runs from directory."""
    if not n_clicks or not directory:
        return {}, [], [], {}
    
    # First, look for unified files (H5 or CSV+NPY)
    unified_files = find_unified_h5_files(directory)
    
    if unified_files:
        # We found unified files - prioritize these
        runs_data = {}
        random_baseline_data = {}
        unified_data = {}
        unique_run_ids = set()
        
        for unified_file in unified_files:
            # Handle both H5 and JSON metadata files
            if unified_file.endswith('.h5'):
                run_id = extract_run_id(unified_file)
                unified_data[run_id] = unified_file
                runs_data[run_id] = {'type': 'unified', 'path': unified_file}
                random_baseline_data[run_id] = {'type': 'unified', 'path': unified_file}
            else:  # It's a metadata JSON file
                try:
                    with open(unified_file, 'r') as f:
                        metadata = json.load(f)
                    run_id = metadata.get('run_id')
                    if run_id:
                        unified_data[run_id] = unified_file
                        runs_data[run_id] = {'type': 'csv_npy', 'path': unified_file}
                        random_baseline_data[run_id] = {'type': 'csv_npy', 'path': unified_file}
                except Exception as e:
                    print(f"Error loading metadata from {unified_file}: {e}")
                    continue
            
            unique_run_ids.add(run_id)
        
        options = [{"label": run_id, "value": run_id} for run_id in sorted(unique_run_ids)]
        
        dataset_info = [
            html.H4("Dataset Info"),
            html.P(f"Found {len(unique_run_ids)} runs in {directory}"),
            html.P(f"Using unified format (H5 or CSV+NPY)")
        ]
        
        return runs_data, options, dataset_info, random_baseline_data
    
    # Look for CSV files next (standalone, without metadata)
    csv_files, random_csv_files = find_csv_npy_files(directory)
    
    if csv_files:
        options = [{"label": run_id, "value": run_id} for run_id in sorted(csv_files.keys())]
        
        dataset_info = [
            html.H4("Dataset Info"),
            html.P(f"Found {len(csv_files)} runs in {directory}"),
            html.P(f"Using CSV format")
        ]
        
        return csv_files, options, dataset_info, random_csv_files
    
    # Original path for legacy CSV files
    csv_files, random_csv_files = find_csv_files(directory)
    
    if csv_files:
        options = [{"label": run_id, "value": run_id} for run_id in sorted(csv_files.keys())]
        
        dataset_info = [
            html.H4("Dataset Info"),
            html.P(f"Found {len(csv_files)} runs in {directory}"),
            html.P(f"Using legacy CSV format")
        ]
        
        return csv_files, options, dataset_info, random_csv_files
    
    # No files found
    return {}, [], [html.H4("No data found"), html.P(f"No data files found in {directory}")], {}

# Callback to load data for selected run
@app.callback(
    [Output('current-df', 'data'),
     Output('singular-values-data', 'data'),
     Output('weights-data', 'data'),
     Output('target-dropdown', 'options'),
     Output('layer-dropdown', 'options'),
     Output('multi-layer-dropdown', 'options'),
     Output('checkpoint-dropdown', 'options'),
     Output('debug-info', 'children')],
    [Input('run-id-dropdown', 'value')],
    [State('runs-data', 'data'),
     State('random-baseline-data', 'data')]
)
def load_run_data(run_id, runs_data, random_baseline_data):
    """Load data for a specific run."""
    # Initialize with empty values
    df = None
    singular_values_data = {}
    weights_data = {}
    target_options = []
    layer_options = []
    multi_layer_options = []
    checkpoint_options = []
    debug_info = [f"Loading data for run {run_id}"]
    
    if not run_id or not runs_data or run_id not in runs_data:
        return None, json.dumps({}), json.dumps({}), [], [], [], [], "No run selected"
    
    run_data = runs_data[run_id]
    
    # Handle unified H5 format
    if isinstance(run_data, dict) and run_data.get('type') == 'unified':
        debug_info.append(f"Loading unified H5 data for run {run_id}")
        unified_file = run_data['path']
        
        try:
            # Load the unified data
            unified_data = load_unified_h5_data(unified_file)
            
            if not unified_data:
                return None, json.dumps({}), json.dumps({}), [], [], [], [], f"Error loading unified data for run {run_id}"
            
            # Extract the main dataframe
            df = unified_data.get('checkpoint_df')
            
            if df is None:
                return None, json.dumps({}), json.dumps({}), [], [], [], [], f"No checkpoint data found in unified file for run {run_id}"
            
            # Add random baseline data if available
            random_df = unified_data.get('random_df')
            if random_df is not None:
                # Add a flag to identify random baseline data (if not already present)
                if 'is_random' not in random_df.columns:
                    random_df['is_random'] = True
                # Combine the dataframes
                df = pd.concat([df, random_df], ignore_index=True)
            
            # Extract singular values and weights
            singular_values_data = unified_data.get('singular_values', {}) or {}
            weights_data = unified_data.get('weights', {}) or {}
            
            debug_info.append(f"Successfully loaded unified data for run {run_id}")
        except Exception as e:
            import traceback
            debug_info.append(f"Error loading unified data: {str(e)}")
            debug_info.append(traceback.format_exc())
            return None, json.dumps({}), json.dumps({}), [], [], [], [], f"Error loading unified data: {str(e)}"
    
    # Handle CSV+NPY format
    elif isinstance(run_data, dict) and run_data.get('type') == 'csv_npy':
        debug_info.append(f"Loading CSV+NPY data for run {run_id}")
        metadata_path = run_data['path']
        base_dir = os.path.dirname(metadata_path)
        
        try:
            # Load the CSV+NPY data
            csv_npy_data = load_csv_npy_data(run_id, base_dir)
            
            if not csv_npy_data:
                return None, json.dumps({}), json.dumps({}), [], [], [], [], f"Error loading CSV+NPY data for run {run_id}"
            
            # Extract the main dataframe
            df = csv_npy_data.get('checkpoint_df')
            
            if df is None:
                return None, json.dumps({}), json.dumps({}), [], [], [], [], f"No checkpoint data found in CSV+NPY data for run {run_id}"
            
            # Add random baseline data if available
            random_df = csv_npy_data.get('random_df')
            if random_df is not None:
                # Add a flag to identify random baseline data (if not already present)
                if 'is_random' not in random_df.columns:
                    random_df['is_random'] = True
                # Combine the dataframes
                df = pd.concat([df, random_df], ignore_index=True)
            
            # Extract singular values and weights
            singular_values_data = csv_npy_data.get('singular_values', {}) or {}
            weights_data = csv_npy_data.get('weights', {}) or {}
            
            # Also include random singular values if available
            random_sv = csv_npy_data.get('random_singular_values', {})
            if random_sv:
                for target, target_sv in random_sv.items():
                    if target not in singular_values_data:
                        singular_values_data[target] = {}
                    for layer, layer_entries in target_sv.items():
                        if layer not in singular_values_data[target]:
                            singular_values_data[target][layer] = []
                        singular_values_data[target][layer].extend(layer_entries)
            
            debug_info.append(f"Successfully loaded CSV+NPY data for run {run_id}")
        except Exception as e:
            import traceback
            debug_info.append(f"Error loading CSV+NPY data: {str(e)}")
            debug_info.append(traceback.format_exc())
            return None, json.dumps({}), json.dumps({}), [], [], [], [], f"Error loading CSV+NPY data: {str(e)}"
    
    # Check if this is a CSV-based dataset
    elif isinstance(run_data, dict) and 'results' in run_data:
        # Handle regular CSV files (existing functionality)
        debug_info.append(f"Loading CSV data for run {run_id}")
        
        try:
            # Load main results
            main_file = run_data['results']
            df = load_csv_dataframe(main_file)
            
            if df is None:
                return None, json.dumps({}), json.dumps({}), [], [], [], [], f"Error loading CSV data for run {run_id}"
            
            # Set flag to identify checkpoint data
            if 'is_random' not in df.columns:
                df['is_random'] = False
            
            # Load singular values
            if 'singular_values' in run_data and run_data['singular_values']:
                singular_values_data = load_singular_values(run_data['singular_values']) or {}
            
            # Load weights
            if 'weights' in run_data and run_data['weights']:
                weights_data = load_weights_dataframe(run_data['weights']) or {}
            
            # Load random baseline if it exists
            if 'random' in run_data and run_data['random'] and os.path.exists(run_data['random']):
                random_df = load_csv_dataframe(run_data['random'])
                if random_df is not None:
                    # Add random dataframe to the regular dataframe
                    if 'is_random' not in random_df.columns:
                        random_df['is_random'] = True
                    df = pd.concat([df, random_df], ignore_index=True)
            
            debug_info.append(f"Successfully loaded CSV data for run {run_id}")
        except Exception as e:
            import traceback
            debug_info.append(f"Error loading CSV data: {str(e)}")
            debug_info.append(traceback.format_exc())
            return None, json.dumps({}), json.dumps({}), [], [], [], [], f"Error loading CSV data: {str(e)}"
    
    # Unknown format
        else:
        debug_info.append(f"Unknown data format for run {run_id}")
        return None, json.dumps({}), json.dumps({}), [], [], [], [], f"Unknown data format for run {run_id}"
    
    # If we have a dataframe, extract targets, layers, and checkpoints
    targets = []
    layers = []
    checkpoints = []
    target_options = []
    layer_options = []
    multi_layer_options = []
    checkpoint_options = []
    
    if df is not None:
        try:
            targets = df['target'].unique().tolist()
            target_options = [{"label": t, "value": t} for t in sorted(targets)]
            
            layers = df['layer_name'].unique().tolist()
            layer_options = [{"label": l, "value": l} for l in sorted(layers)]
            multi_layer_options = layer_options  # Same options for multi-layer dropdown
            
            checkpoints = df['checkpoint'].unique().tolist()
            checkpoint_options = [{"label": str(c), "value": str(c)} for c in sorted(checkpoints)]
            
            success_message = f"Loaded data for run {run_id} with {len(targets)} targets, {len(layers)} layers, {len(checkpoints)} checkpoints"
            debug_info.append(success_message)
            
            return (
                df.to_json(orient='split'),
                json.dumps(singular_values_data),
                json.dumps(weights_data),
                target_options,
                layer_options,
                multi_layer_options,
                checkpoint_options,
                "\n".join(debug_info)
            )
    except Exception as e:
            import traceback
            debug_info.append(f"Error processing dataframe: {str(e)}")
            debug_info.append(traceback.format_exc())
            return None, json.dumps({}), json.dumps({}), [], [], [], [], f"Error processing dataframe: {str(e)}"
        else:
        return None, json.dumps({}), json.dumps({}), [], [], [], [], "No valid data found for the selected run"

# Callback to update Checkpoint Evolution
@app.callback(
    Output('checkpoint-evolution', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('current-df', 'data'),
     State('target-dropdown', 'value'),
     State('layer-dropdown', 'value'),
     State('multi-layer-dropdown', 'value'),  # Add multi-layer input
     State('rcond-dropdown', 'value'),
     State('metric-selector', 'value'),
     State('distance-normalization', 'value')]  # Add normalization state
)
def update_checkpoint_evolution(n_clicks, df_json, target, layer, multi_layers, rcond, metric, normalization='none'):
    if not df_json or not target:
        return go.Figure().update_layout(title="No data selected")
    
    try:
        df = json_to_dataframe(df_json)
        
        # Get metric information
        metric_info = get_metric_info(metric)
        
        # Handle single layer or multiple layers
        if multi_layers and isinstance(multi_layers, list) and len(multi_layers) > 0:
            layers = multi_layers
        else:
            layers = [layer] if layer else []
        
        if not layers:
            return go.Figure().update_layout(title="No layers selected")
        
        # Filter for the target
        target_filter = df['target'] == target
        if not any(target_filter):
            return go.Figure().update_layout(title=f"No data found for target={target}")
        
        # Identify non-random (trained) vs random checkpoints
        non_random_filter = df['is_random'] == False
        random_filter = df['is_random'] == True
        
        # Filter data for analysis
        trained_df = df[target_filter & non_random_filter]
        random_df = df[target_filter & random_filter]
        
        if trained_df.empty:
            return go.Figure().update_layout(title="No trained checkpoint data found")
        
        # Check if using best rcond or a specific value
        using_best_rcond = rcond.lower() == 'best'
        
        # Set normalization factor (initialize to 1.0)
        normalization_factor = {}
        for current_layer in layers:
            normalization_factor[current_layer] = 1.0
        
        # Calculate normalization factors if requested
        if normalization == 'baseline' and 'dist' in metric and not random_df.empty:
            for current_layer in layers:
                layer_random_df = random_df[random_df['layer_name'] == current_layer]
                if not layer_random_df.empty:
                    mean_val = layer_random_df[metric].mean()
                    if mean_val > 0:
                        normalization_factor[current_layer] = mean_val
                        print(f"Normalizing {current_layer} by random baseline: {mean_val}")
        
        # Create figure
        fig = go.Figure()
        
        # Extract checkpoint step numbers for sorting
        def extract_checkpoint_number(checkpoint_name):
            try:
                # Look for a number at the end of the checkpoint name
                # This assumes checkpoints have format like "checkpoint_1000"
                if 'RANDOM' in str(checkpoint_name):
                    return -1  # Place random at the beginning
                # Try to extract different formats of checkpoint numbers
                if '_' in str(checkpoint_name):
                    parts = str(checkpoint_name).split('_')
                    for part in parts:
                        if part.isdigit():
                            return int(part)
                return 0  # Default value
            except:
                return 0  # Default value
        
        # Process each layer
        for current_layer in layers:
            # Filter data for this layer
            layer_df = trained_df[trained_df['layer_name'] == current_layer]
            
            if layer_df.empty:
                continue
            
            if using_best_rcond:
                # Find best rcond for each checkpoint
                results = []
                
                for checkpoint in layer_df['checkpoint'].unique():
                    best_rcond_val = find_best_rcond(df, target, checkpoint, current_layer, metric, metric_info['better'])
                    
                    if best_rcond_val is not None:
                        checkpoint_df = layer_df[
                            (layer_df['checkpoint'] == checkpoint) &
                            np.isclose(layer_df['rcond'], best_rcond_val, rtol=1e-5, atol=1e-12)
                        ]
                        
                        if not checkpoint_df.empty:
                            results.append(checkpoint_df)
                
                if not results:
                    continue
                
                filtered_layer_df = pd.concat(results, ignore_index=True)
            else:
                # Filter for specific rcond
                rcond_filter = np.isclose(layer_df['rcond'], float(rcond), rtol=1e-5, atol=1e-12)
                filtered_layer_df = layer_df[rcond_filter]
                
                if filtered_layer_df.empty:
                    continue
            
            # Extract unique checkpoints and sort them
            checkpoints = sorted(filtered_layer_df['checkpoint'].unique(), key=extract_checkpoint_number)
            
            # Aggregate metrics for each checkpoint
            checkpoint_metrics = []
            
            for checkpoint in checkpoints:
                checkpoint_df = filtered_layer_df[filtered_layer_df['checkpoint'] == checkpoint]
                
                if checkpoint_df.empty:
                    continue
                
                # Calculate normalized metric if needed
                metric_value = checkpoint_df[metric].mean()
                if normalization == 'baseline' and 'dist' in metric and normalization_factor[current_layer] > 0:
                    metric_value = metric_value / normalization_factor[current_layer]
                
                checkpoint_metrics.append({
                    'checkpoint': checkpoint,
                    'metric': metric_value,
                    'number': extract_checkpoint_number(checkpoint)
                })
            
            if not checkpoint_metrics:
                continue
            
            # Sort by checkpoint number
            checkpoint_metrics.sort(key=lambda x: x['number'])
            
            # Extract data for plotting
            x_values = [item['number'] for item in checkpoint_metrics if item['number'] > 0]
            y_values = [item['metric'] for item in checkpoint_metrics if item['number'] > 0]
            
            # Only plot if we have data points
            if not x_values:
                continue
            
            # Add trace for this layer
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name=current_layer,
                    line=dict(width=2),
                    marker=dict(size=8)
                )
            )
            
            # Add random baseline KDE if available
            if not random_df.empty and normalization != 'baseline':
                layer_random_df = random_df[random_df['layer_name'] == current_layer]
                
                if not layer_random_df.empty:
                    # Calculate distribution statistics for the random baseline
                    random_values = layer_random_df[metric].values
                    random_mean = np.mean(random_values)
                    random_std = np.std(random_values)
                    
                    # Generate KDE y-values for the distribution (simplified representation)
                    kde_y = np.linspace(
                        max(0, random_mean - 2*random_std),
                        random_mean + 2*random_std,
                        20
                    )
                    
                    # Generate simplified normal distribution for the KDE
                    from scipy.stats import norm
                    kde_x = norm.pdf(kde_y, random_mean, random_std)
                    # Scale the KDE x values to fit nicely on the plot
                    max_kde_x = max(kde_x)
                    scale_factor = (min(x_values) - 0) / max_kde_x * 0.8
                    kde_x = [min(x_values) - x * scale_factor for x in kde_x]
                    
                    # Add the KDE trace
                    fig.add_trace(
                        go.Scatter(
                            x=kde_x,
                            y=kde_y,
                            mode='lines',
                            line=dict(color=fig.data[-1].line.color, width=1.5),
                            name=f"{current_layer} (Random)",
                            fill='tozerox',
                            fillcolor=fig.data[-1].line.color.replace('rgb', 'rgba').replace(')', ', 0.3)'),
                        )
                    )
                    
                    # Add a marker for the random mean
                    fig.add_trace(
                        go.Scatter(
                            x=[min(x_values) - 0.5],
                            y=[random_mean],
                            mode='markers',
                            marker=dict(
                                color=fig.data[-2].line.color,
                                size=10,
                                symbol='diamond'
                            ),
                            name=f"{current_layer} Random (mean)"
                        )
                    )
            
            # If we're using normalization, add a reference line at y=1
            if normalization == 'baseline':
                fig.add_shape(
                    type="line",
                    x0=min(x_values),
                    x1=max(x_values),
                    y0=1,
                    y1=1,
                    line=dict(
                        color="rgba(0, 0, 0, 0.5)",
                        width=1,
                        dash="dash",
                    )
                )
        
        # Update layout
        rcond_display = "best" if using_best_rcond else f"{rcond:.2e}"
        metric_display = metric_info['title']
        
        if normalization == 'baseline' and 'dist' in metric:
            metric_display += " (normalized by random baseline)"
        
        fig.update_layout(
            title=f"Training Dynamics for {target} (rcond={rcond_display})",
            xaxis_title="Training Step",
            yaxis_title=metric_display,
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=60, r=50, b=60, l=60),
            hovermode="closest"
        )
        
        return fig
        
    except Exception as e:
        import traceback
        return go.Figure().update_layout(
            title="Error updating chart",
            annotations=[{
                'text': f"Error: {str(e)}\n{traceback.format_exc()}",
                'showarrow': False,
                'x': 0.5,
                'y': 0.5
            }]
        )

# Callback to update R² vs Distance Plot
@app.callback(
    Output('r-squared-vs-dist', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('current-df', 'data'),
     State('target-dropdown', 'value'),
     State('checkpoint-dropdown', 'value'),
     State('rcond-dropdown', 'value'),
     State('distance-normalization', 'value')]  # Add normalization state
)
def update_rsquared_vs_dist(n_clicks, df_json, target, checkpoints, rcond, normalization='none'):
    if not df_json or not target or not checkpoints:
        return go.Figure().update_layout(title="No data selected")
    
    try:
        df = json_to_dataframe(df_json)
        
        # Ensure checkpoint is a list
        if not isinstance(checkpoints, list):
            checkpoints = [checkpoints]
        
        # Filter for the target and checkpoints
        target_filter = df['target'] == target
        checkpoint_filter = df['checkpoint'].isin(checkpoints)
        
        # Filter for non-random checkpoints (exclude random baselines)
        non_random_filter = df['is_random'] == False
        
        # Combined filter
        base_filter = target_filter & checkpoint_filter & non_random_filter
        
        if not any(base_filter):
            checkpoint_list = ", ".join(checkpoints)
            msg = f"No data found for target='{target}', checkpoint(s)={checkpoint_list}"
            return go.Figure().update_layout(title=msg)
        
        # Check if using best rcond or a specific value
        using_best_rcond = rcond.lower() == 'best'
        
        if using_best_rcond:
            # For each layer, find the best rcond based on norm_dist
            best_rows = []
            
            for layer in df[base_filter]['layer_name'].unique():
                for checkpoint in checkpoints:
                    best_rcond_val = find_best_rcond(df, target, checkpoint, layer, 'norm_dist', 'min')
                    
                    if best_rcond_val is not None:
                        best_row = df[
                            (df['target'] == target) &
                            (df['checkpoint'] == checkpoint) &
                            (df['layer_name'] == layer) &
                            np.isclose(df['rcond'], best_rcond_val, rtol=1e-5, atol=1e-12)
                        ]
                        
                        if not best_row.empty:
                            best_rows.append(best_row)
            
            if not best_rows:
                checkpoint_list = ", ".join(checkpoints)
                msg = f"No best rcond data found for target='{target}', checkpoint(s)={checkpoint_list}"
                return go.Figure().update_layout(title=msg)
            
            filtered_df = pd.concat(best_rows, ignore_index=True)
        else:
            # Filter for a specific rcond
            rcond_filter = np.isclose(df['rcond'], float(rcond), rtol=1e-5, atol=1e-12)
            filtered_df = df[base_filter & rcond_filter]
            
            if filtered_df.empty:
                rcond_display = f"{rcond:.2e}"
                checkpoint_list = ", ".join(checkpoints)
                msg = f"No data found with target='{target}', checkpoint(s)={checkpoint_list}, rcond={rcond_display}"
                return go.Figure().update_layout(title=msg)
        
        # Apply normalization if requested
        norm_dist_column = "norm_dist"
        if normalization == 'baseline' and not df[df['is_random'] == True].empty:
            # Get random baseline data for normalization
            random_df = df[df['is_random'] == True & target_filter]
            
            if not random_df.empty:
                # Calculate mean norm_dist by layer for random baseline
                layer_means = random_df.groupby('layer_name')['norm_dist'].mean().to_dict()
                
                # Create normalized column
                filtered_df['norm_dist_normalized'] = filtered_df.apply(
                    lambda row: row['norm_dist'] / layer_means.get(row['layer_name'], 1.0) 
                    if row['layer_name'] in layer_means and layer_means[row['layer_name']] > 0 
                    else row['norm_dist'],
                    axis=1
                )
                
                # Use the normalized column
                norm_dist_column = "norm_dist_normalized"
        
        # Create a scatter plot
        rcond_display = "best" if using_best_rcond else f"{rcond:.2e}"
        
        # Adjust labels based on normalization
        x_axis_label = "Normalized Distance (relative to random baseline)" if norm_dist_column == "norm_dist_normalized" else "Normalized Distance (lower is better)"
        
        fig = px.scatter(
            filtered_df, 
            x=norm_dist_column, 
            y="r_squared",
            color="layer_name",
            symbol="checkpoint",
            size="dims",  # Size represents the dimensionality
            hover_data=["layer_name", "checkpoint", "dims", "rcond"],
            title=f"R² vs. Distance for {target} (rcond={rcond_display})",
            labels={
                norm_dist_column: x_axis_label, 
                "r_squared": "R² (higher is better)",
                "dims": "Dimensionality"
            },
            height=500
        )
        
        # Add a trendline if there are enough points
        if len(filtered_df) > 1:
            z = np.polyfit(filtered_df[norm_dist_column], filtered_df['r_squared'], 1)
            p = np.poly1d(z)
            
            x_range = np.linspace(filtered_df[norm_dist_column].min(), filtered_df[norm_dist_column].max(), 100)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name='Trend'
                )
            )
        
        # Add a vertical line at x=1 if normalized (meaning equal to random baseline)
        if norm_dist_column == "norm_dist_normalized":
            fig.add_shape(
                type="line",
                x0=1, x1=1,
                y0=0, y1=1,
                line=dict(
                    color="rgba(0, 0, 0, 0.5)",
                    width=1,
                    dash="dash",
                ),
                yref="paper"
            )
            
            fig.add_annotation(
                x=1,
                y=0.02,
                text="Random baseline level",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                yref="paper"
            )
        
        # Improve layout
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=60, r=50, b=60, l=60),
        )
        
        return fig
        
    except Exception as e:
        import traceback
        return go.Figure().update_layout(
            title="Error updating chart",
            annotations=[{
                'text': f"Error: {str(e)}\n{traceback.format_exc()}",
                'showarrow': False,
                'x': 0.5,
                'y': 0.5
            }]
        )

# Callback to update Singular Value Plot
@app.callback(
    Output('singular-value-plot', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('singular-values-data', 'data'),
     State('target-dropdown', 'value'),
     State('layer-dropdown', 'value'),
     State('checkpoint-dropdown', 'value')]
)
def update_singular_value_plot(n_clicks, singular_values, target, layer, checkpoints):
    if not singular_values or not target or not layer:
        return go.Figure().update_layout(title="No data selected")
    
    try:
        # Check if the target and layer exist in the data
        if target not in singular_values or layer not in singular_values[target]:
            return go.Figure().update_layout(
                title=f"No singular values available for {layer} in {target}",
                height=500
            )
        
        # Get the singular values
        sv_data = singular_values[target][layer]
        
        # Filter to selected checkpoints if provided
        if checkpoints:
            if isinstance(checkpoints, list):
                checkpoint_list = [str(c) for c in checkpoints]
                sv_data = [item for item in sv_data if 'checkpoint' in item and item['checkpoint'] in checkpoint_list]
            else:
                checkpoint = str(checkpoints)
                sv_data = [item for item in sv_data if 'checkpoint' in item and item['checkpoint'] == checkpoint]
        
        if not sv_data:
            return go.Figure().update_layout(
                title=f"No singular values available for selected checkpoints",
                height=500
            )
        
        # Create a figure
        fig = go.Figure()
        
        # Add a trace for each checkpoint
        for item in sv_data:
            sv = item['singular_values']
            
            # Get checkpoint or random index for the label
            if 'checkpoint' in item:
                label = f"Checkpoint {item['checkpoint']}"
            elif 'random_idx' in item:
                label = f"Random {item['random_idx']}"
            else:
                label = "Unknown"
            
            # Add the trace
            fig.add_trace(
                go.Scatter(
                    y=sv,
                    mode='lines',
                    name=label
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"Singular Values for {layer} in {target}",
            xaxis_title="Component Index",
            yaxis_title="Singular Value",
            yaxis_type="log",  # Log scale often better for viewing singular values
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=60, r=50),
            hovermode="closest",
            height=500
        )
        
        return fig
    except Exception as e:
        return go.Figure().update_layout(
            title=f"Error updating chart: {str(e)}",
            height=500
        )

# Callback to update Cumulative Variance Plot
@app.callback(
    Output('cumulative-variance', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('current-df', 'data'),
     State('target-dropdown', 'value'),
     State('layer-dropdown', 'value'),
     State('checkpoint-dropdown', 'value')]
)
def update_cumulative_variance(n_clicks, df_json, target, layer, checkpoints):
    if not df_json or not target or not layer:
        return go.Figure().update_layout(title="No data selected")
    
    try:
        # Convert JSON to DataFrame
        df = json_to_dataframe(df_json)
        
        # Ensure proper types for comparison
        df['checkpoint_str'] = df['checkpoint'].apply(debug_serialize)
        
        # For simplicity, use the first selected checkpoint
        if isinstance(checkpoints, list) and len(checkpoints) > 0:
            checkpoint = str(checkpoints[0])
        else:
            checkpoint = str(checkpoints)
        
        # Filter the dataframe
        filtered_df = df[
            (df['target'] == target) & 
            (df['layer_name'] == layer) &
            (df['checkpoint_str'] == checkpoint)
        ]
        
        if filtered_df.empty:
            msg = f"No data found with target='{target}', layer='{layer}', checkpoint='{checkpoint}'"
            return go.Figure().update_layout(
                title=msg,
                height=500
            )
        
        # Take the first row since all rows for the same layer/target/checkpoint should have the same variance values
        row = filtered_df.iloc[0]
        
        # Extract variance explained data with better error handling
        var_expl = None
        
        # First, let's check if the column exists
        if 'variance_explained' not in row:
            return go.Figure().update_layout(
                title="'variance_explained' column not found in the data",
                height=500
            )
        
        var_expl_raw = row['variance_explained']
        var_expl_type = type(var_expl_raw).__name__
        
        # Safe parsing of variance_explained
        try:
            # If it's already a list or numpy array, use it directly
            if isinstance(var_expl_raw, (list, tuple, np.ndarray)):
                var_expl = list(var_expl_raw)
            # If it's a string, try multiple parsing methods
            elif isinstance(var_expl_raw, str):
                # First clean up the string
                var_expl_str = var_expl_raw.strip()
                
                # Check for empty string
                if not var_expl_str:
                    return go.Figure().update_layout(
                        title="Empty variance_explained value",
                        height=500
                    )
                
                # Check for unbalanced brackets (missing closing bracket)
                if var_expl_str.count('[') > var_expl_str.count(']'):
                    # Add missing closing bracket
                    var_expl_str = var_expl_str + ']'
                
                # Try JSON first (most reliable)
                try:
                    var_expl = json.loads(var_expl_str)
                except json.JSONDecodeError:
                    # If JSON fails, try ast.literal_eval (safer than eval)
                    try:
                        import ast
                        var_expl = ast.literal_eval(var_expl_str)
                    except (SyntaxError, ValueError):
                        # If all parsing methods fail, try to split by commas if it looks like a list
                        if var_expl_str.startswith('[') and var_expl_str.endswith(']'):
                            try:
                                var_expl = [float(x.strip()) for x in var_expl_str.strip('[]').split(',') if x.strip()]
                            except ValueError:
                                return go.Figure().update_layout(
                                    title=f"Could not parse variance_explained as list: {var_expl_str[:50]}...",
                                    height=500,
                                    annotations=[{
                                        'text': f"Type: {var_expl_type}, Value: {var_expl_str[:100]}...",
                                        'showarrow': False,
                                        'x': 0.5,
                                        'y': 0.5
                                    }]
                                )
                        else:
                            return go.Figure().update_layout(
                                title=f"Variance explained is not in a recognized format",
                                annotations=[{
                                    'text': f"Type: {var_expl_type}, Value: {var_expl_str[:100]}...",
                                    'showarrow': False,
                                    'x': 0.5,
                                    'y': 0.5
                                }],
                                height=500
                            )
            else:
                # For other types, try direct conversion if possible
                try:
                    var_expl = list(var_expl_raw)
                except:
                    return go.Figure().update_layout(
                        title=f"Variance explained could not be converted to list",
                        annotations=[{
                            'text': f"Type: {var_expl_type}, Value: {str(var_expl_raw)[:100]}...",
                            'showarrow': False,
                            'x': 0.5,
                            'y': 0.5
                        }],
                        height=500
                    )
                
        except Exception as e:
            return go.Figure().update_layout(
                title=f"Error parsing variance explained data: {str(e)}",
                annotations=[{
                    'text': f"Raw data type: {var_expl_type}, value: {str(var_expl_raw)[:100]}...",
                    'showarrow': False,
                    'x': 0.5,
                    'y': 0.5
                }],
                height=500
            )
        
        # Check if parsing was successful
        if not var_expl:
            return go.Figure().update_layout(
                title="No variance explained data available after parsing",
                annotations=[{
                    'text': f"Raw data type: {var_expl_type}, value: {str(var_expl_raw)[:100]}...",
                    'showarrow': False,
                    'x': 0.5,
                    'y': 0.5
                }],
                height=500
            )
        
        # Convert all elements to float
        try:
            var_expl = [float(x) for x in var_expl]
        except (ValueError, TypeError) as e:
            return go.Figure().update_layout(
                title=f"Error converting values to float: {str(e)}",
                annotations=[{
                    'text': f"Parsed data: {var_expl[:10]}...",
                    'showarrow': False,
                    'x': 0.5,
                    'y': 0.5
                }],
                height=500
            )
        
        # Create a figure
        fig = go.Figure()
        
        # Add the cumulative variance curve
        fig.add_trace(
            go.Scatter(
                y=var_expl,
                mode='lines',
                name='Cumulative Variance',
                line=dict(color='blue')
            )
        )
        
        # Add reference lines for 80%, 90%, 95%, 99%
        reference_levels = [0.8, 0.9, 0.95, 0.99]
        
        for level in reference_levels:
            # Find the index where we exceed this level
            try:
                idx = next(i for i, x in enumerate(var_expl) if x >= level)
                
                # Add a horizontal line
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=level,
                    x1=len(var_expl),
                    y1=level,
                    line=dict(
                        color="red",
                        width=1,
                        dash="dash",
                    )
                )
                
                # Add a vertical line
                fig.add_shape(
                    type="line",
                    x0=idx,
                    y0=0,
                    x1=idx,
                    y1=level,
                    line=dict(
                        color="red",
                        width=1,
                        dash="dash",
                    )
                )
                
                # Add annotation
                fig.add_annotation(
                    x=idx,
                    y=level,
                    text=f"{level*100:.0f}% at {idx+1} components",
                    showarrow=True,
                    arrowhead=1
                )
            except StopIteration:
                continue
        
        # Update layout
        fig.update_layout(
            title=f"Cumulative Variance Explained for {layer} in {target} (Checkpoint {checkpoint})",
            xaxis_title="Number of Components",
            yaxis_title="Cumulative Variance Explained",
            yaxis=dict(range=[0, 1.05]),
            margin=dict(t=60, r=50),
            hovermode="closest",
            height=500
        )
        
        return fig
    except Exception as e:
        return go.Figure().update_layout(
            title=f"Error updating chart: {str(e)}",
            annotations=[{
                'text': str(e),
                'showarrow': False,
                'x': 0.5,
                'y': 0.5
            }],
            height=500
        )

# Callback to update Random Baseline Comparison
@app.callback(
    Output('random-baseline-comparison', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('current-df', 'data'),
     State('target-dropdown', 'value'),
     State('layer-dropdown', 'value'),
     State('rcond-dropdown', 'value'),
     State('metric-selector', 'value'),
     State('distance-normalization', 'value')]
)
def update_random_baseline(n_clicks, df_json, target, layer, rcond, metric, normalization):
    if not df_json or not target or not layer:
        return go.Figure().update_layout(title="No data or selections available")
    
    df = json_to_dataframe(df_json)
    
    metric_info = get_metric_info(metric)
    metric_title = metric_info['title']  # Use 'title' instead of 'label'
    better = metric_info['better']
    
    # Check that we have random baselines
    random_filter = df['is_random'] == True
    trained_filter = df['is_random'] == False
    
    if not any(random_filter) or not any(trained_filter):
        return go.Figure().update_layout(
            title="No random baseline data available",
            annotations=[{
                'text': "Cannot find random baseline data in the dataset. Random baseline requires checkpoint names containing 'RANDOM'.",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
    
    # Filter data for the target and layer
    target_layer_filter = (df['target'] == target) & (df['layer_name'] == layer)
    random_base_df = df[random_filter & target_layer_filter]
    trained_base_df = df[trained_filter & target_layer_filter]
    
    if random_base_df.empty or trained_base_df.empty:
        return go.Figure().update_layout(
            title=f"No data available for target={target}, layer={layer}",
            annotations=[{
                'text': f"Cannot find data for both trained models and random baselines with target={target}, layer={layer}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
    
    using_best_rcond = rcond.lower() == 'best'
    
    # Apply normalization if requested
    normalization_factor = 1.0
    if normalization == 'baseline' and 'dist' in metric:
        # Calculate mean of random baseline models for normalization
        mean_baseline = random_base_df[metric].mean()
        if mean_baseline > 0:
            normalization_factor = mean_baseline
            # Log the normalization for debugging
            print(f"Normalizing by mean baseline: {mean_baseline}")
    
    # Set up the figure
    fig = go.Figure()
    
    # Results holders
    trained_df = pd.DataFrame()
    random_df = pd.DataFrame()
    
    try:
        if using_best_rcond:
            # For trained models
            trained_rows = []
            trained_checkpoints = trained_base_df['checkpoint'].unique().tolist()
            
            for checkpoint in trained_checkpoints:
                best_rcond_val = find_best_rcond(df, target, checkpoint, layer, metric, better)
                
                if best_rcond_val is not None:
                    best_row = df[
                        (df['target'] == target) &
                        (df['checkpoint'] == checkpoint) &
                        (df['layer_name'] == layer) &
                        np.isclose(df['rcond'], best_rcond_val, rtol=1e-5, atol=1e-12)
                    ]
                    
                    if not best_row.empty:
                        trained_rows.append(best_row)
            
            trained_df = pd.concat(trained_rows, ignore_index=True) if trained_rows else pd.DataFrame()
            
            # For random models
            random_rows = []
            random_checkpoints = random_base_df['checkpoint'].unique().tolist()
            
            for checkpoint in random_checkpoints:
                best_rcond_val = find_best_rcond(df, target, checkpoint, layer, metric, better)
                
                if best_rcond_val is not None:
                    best_row = df[
                        (df['target'] == target) &
                        (df['checkpoint'] == checkpoint) &
                        (df['layer_name'] == layer) &
                        np.isclose(df['rcond'], best_rcond_val, rtol=1e-5, atol=1e-12)
                    ]
                    
                    if not best_row.empty:
                        random_rows.append(best_row)
            
            random_df = pd.concat(random_rows, ignore_index=True) if random_rows else pd.DataFrame()
                
        else:
            # Filter for a specific rcond
            rcond_filter = np.isclose(df['rcond'], float(rcond), rtol=1e-5, atol=1e-12)
            
            trained_df = trained_base_df[rcond_filter]
            random_df = random_base_df[rcond_filter]
        
        if trained_df.empty or random_df.empty:
            rcond_display = "best" if using_best_rcond else f"{rcond:.2e}"
            msg = f"Insufficient data for comparison with rcond={rcond_display}"
            return go.Figure().update_layout(
                title=msg,
                annotations=[{
                    'text': f"No data for both random and trained models with target={target}, layer={layer}, rcond={rcond_display}",
                    'showarrow': False,
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5
                }]
            )
        
        # Apply normalization to metric values
        trained_metrics = trained_df[metric].values
        random_metrics = random_df[metric].values
        
        if normalization == 'baseline' and 'dist' in metric and normalization_factor > 0:
            trained_metrics = trained_metrics / normalization_factor
            random_metrics = random_metrics / normalization_factor
        
        # Add violin plots
        fig.add_trace(go.Violin(
            y=random_metrics,
            name='Random Baseline',
            box_visible=True,
            meanline_visible=True,
            marker_color='rgba(100, 100, 200, 0.5)',
            line_color='rgba(50, 50, 180, 1)',
        ))
        
        fig.add_trace(go.Violin(
            y=trained_metrics,
            name='Trained Models',
            box_visible=True,
            meanline_visible=True,
            marker_color='rgba(200, 100, 100, 0.5)',
            line_color='rgba(180, 50, 50, 1)',
        ))
        
        # Update layout
        metric_display = f"{metric_title}"  # Use metric_title instead of metric_name
        if normalization == 'baseline' and 'dist' in metric:
            metric_display += f" (normalized by mean baseline)"
        
        rcond_display = "best" if using_best_rcond else f"{rcond:.2e}"
        
        fig.update_layout(
            title=f"Random Baseline vs. Trained Models: {metric_display}",
            xaxis_title="Model Type",
            yaxis_title=metric_display,
            violinmode='group',
            height=600,
            annotations=[{
                'text': f"Target: {target}, Layer: {layer}, rcond: {rcond_display}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 1.05
            }]
        )
        
        # Add mean lines
        random_mean = random_metrics.mean()
        trained_mean = trained_metrics.mean()
        
        fig.update_layout(
            shapes=[
                # Mean line for random
                dict(
                    type="line",
                    x0=-0.3, x1=0.3,
                    y0=random_mean, y1=random_mean,
                    line=dict(color="rgba(50, 50, 180, 1)", width=2, dash="dash"),
                    xref="x", yref="y",
                ),
                # Mean line for trained
                dict(
                    type="line",
                    x0=0.7, x1=1.3,
                    y0=trained_mean, y1=trained_mean,
                    line=dict(color="rgba(180, 50, 50, 1)", width=2, dash="dash"),
                    xref="x", yref="y"
                )
            ]
        )
        
        # Add annotations for mean values
        fig.add_annotation(
            x=0,
            y=random_mean,
            text=f"Mean: {random_mean:.3f}",
            showarrow=True,
            arrowhead=1,
            yshift=10
        )
        
        fig.add_annotation(
            x=1,
            y=trained_mean,
            text=f"Mean: {trained_mean:.3f}",
            showarrow=True,
            arrowhead=1,
            yshift=10
        )
        
        # Add debug info on data sources
        fig.update_layout(
            annotations=fig.layout.annotations + (
                {
                    'text': f"Random Models: {len(random_df)} data points, Trained Models: {len(trained_df)} data points",
                    'showarrow': False,
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': -0.15
                },
            )
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().update_layout(
            title="Error generating comparison",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'xref': 'paper', 
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )

# Add a new helper function for safe debug serialization
def debug_serialize(value):
    """Helper function to safely convert values to string for debugging."""
    if value is None:
        return "None"
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return str(value)
    if isinstance(value, np.ndarray):
        return str(value)
    return str(value)

# Add a new callback for the diagnostics button
@app.callback(
    Output('debug-output', 'children'),
    [Input('run-diagnostics-button', 'n_clicks')],
    [State('runs-data', 'data'),
     State('random-baseline-data', 'data'),
     State('current-df', 'data'),
     State('target-dropdown', 'value'),
     State('layer-dropdown', 'value'),
     State('checkpoint-dropdown', 'value')]
)
def run_diagnostics(n_clicks, runs_data, random_baseline_data, df_json, target, layer, checkpoints):
    if n_clicks == 0:
        return "Click 'Run Diagnostics' to analyze data issues."
    
    debug_info = []
    
    debug_info.append("=== RANDOM BASELINE DIAGNOSTICS ===")
    
    # Check for random baseline files
    debug_info.append(f"Random baseline files found: {len(random_baseline_data) if random_baseline_data else 0}")
    if random_baseline_data:
        debug_info.append("Random baseline file paths:")
        for run_id, file_path in list(random_baseline_data.items())[:3]:
            debug_info.append(f"  - {run_id}: {file_path}")
        if len(random_baseline_data) > 3:
            debug_info.append(f"  - ... ({len(random_baseline_data) - 3} more)")
    
    # Analyze current dataframe if available
    if df_json:
        try:
            df = json_to_dataframe(df_json)
            
            debug_info.append(f"\nDataFrame shape: {df.shape}")
            debug_info.append(f"Column names: {list(df.columns)}")
            
            # Display first few rows to see data structure
            debug_info.append("\nFirst 5 rows of DataFrame:")
            pd_str = StringIO()
            df.head().to_string(pd_str)
            debug_info.append(pd_str.getvalue())
            
            # Check how random checkpoints are identified
            debug_info.append("\nRandom baseline file identification analysis:")
            
            # Show the filenames for reference
            if random_baseline_data:
                debug_info.append("Random baseline filenames (without path):")
                for run_id, file_path in list(random_baseline_data.items())[:3]:
                    filename = os.path.basename(file_path)
                    debug_info.append(f"  - {filename}")
            
            # Check for run_id relationships
            debug_info.append("\nRun ID analysis:")
            debug_info.append(f"Unique run_ids in DataFrame: {df['run_id'].unique().tolist()}")
            debug_info.append(f"Checking if random baseline run_ids are in DataFrame:")
            if random_baseline_data:
                for run_id in random_baseline_data.keys():
                    run_id_short = run_id.split('_random_baseline')[0] if '_random_baseline' in run_id else run_id
                    debug_info.append(f"  - Run ID '{run_id}' (short: '{run_id_short}') exists in DataFrame: {run_id_short in df['run_id'].values}")
            
            # Check for RANDOM checkpoints - explicitly convert to string first
            df['checkpoint_str'] = df['checkpoint'].apply(debug_serialize)
            random_pattern = 'RANDOM'
            random_checkpoints = df[df['checkpoint_str'].str.contains(random_pattern, case=False, na=False)]
            debug_info.append(f"\nRandom checkpoints in DataFrame: {len(random_checkpoints)}")
            
            if not random_checkpoints.empty:
                unique_random_checkpoints = random_checkpoints['checkpoint_str'].unique()
                debug_info.append(f"Unique random checkpoint values: {[str(x) for x in unique_random_checkpoints]}")
                
                # Analyze regex match
                debug_info.append("\nTesting regex pattern on random checkpoints:")
                strict_pattern = r'RANDOM_\d+'
                for idx, ckpt in enumerate(unique_random_checkpoints[:5]):
                    matches_pattern = bool(pd.Series([str(ckpt)]).str.match(strict_pattern)[0])
                    debug_info.append(f"  - '{ckpt}' matches 'RANDOM_\\d+': {matches_pattern}")
                
                # For the current selection, check why random data might not be showing
                if target and layer:
                    random_target_layer = random_checkpoints[
                        (random_checkpoints['target'] == target) & 
                        (random_checkpoints['layer_name'] == layer)
                    ]
                    debug_info.append(f"\nRandom checkpoints for {target}/{layer}: {len(random_target_layer)}")
                    if not random_target_layer.empty:
                        debug_info.append("Sample rows:")
                        for i, row in random_target_layer.head(3).iterrows():
                            is_match = bool(pd.Series([str(row['checkpoint_str'])]).str.match(strict_pattern)[0])
                            debug_info.append(f"  - checkpoint: '{row['checkpoint_str']}', is_random match: {is_match}")
            else:
                debug_info.append("No random checkpoints found in DataFrame.")
                # Add sample of checkpoint values to help diagnose
                sample_ckpts = df['checkpoint_str'].unique()[:10]
                debug_info.append(f"Sample checkpoint values: {[str(x) for x in sample_ckpts]}")
                
                # More detailed analysis of h5 files
                debug_info.append("\nAnalyzing h5 file contents directly:")
                import h5py
                for run_id, file_path in list(random_baseline_data.items())[:1]:  # Just check the first one
                    try:
                        with h5py.File(file_path, 'r') as f:
                            if 'regression_results' in f:
                                results_group = f['regression_results']
                                if 'checkpoint' in results_group:
                                    checkpoints = results_group['checkpoint'][:]
                                    debug_info.append(f"  - First 5 checkpoint values in h5 file: {[str(x) for x in checkpoints[:5]]}")
                                    debug_info.append(f"  - Checkpoint types: {[type(x).__name__ for x in checkpoints[:5]]}")
                                if 'is_random' in results_group:
                                    debug_info.append(f"  - File has 'is_random' column")
                                else:
                                    debug_info.append(f"  - File does not have 'is_random' column")
                    except Exception as e:
                        debug_info.append(f"  - Error reading h5 file: {str(e)}")
                
        except Exception as e:
            debug_info.append(f"Error analyzing DataFrame: {str(e)}")
            import traceback
            debug_info.append(traceback.format_exc())
    
    debug_info.append("\n=== CUMULATIVE VARIANCE DIAGNOSTICS ===")
    
    # Check variance_explained format if target and layer are selected
    if df_json and target and layer and checkpoints:
        try:
            df = json_to_dataframe(df_json)
            
            # For simplicity, use the first selected checkpoint
            if isinstance(checkpoints, list) and len(checkpoints) > 0:
                checkpoint = str(checkpoints[0])
            else:
                checkpoint = str(checkpoints)
            
            # Filter the dataframe - ensure checkpoint is string for comparison
            df['checkpoint_str'] = df['checkpoint'].apply(debug_serialize)
            filtered_df = df[
                (df['target'] == target) & 
                (df['layer_name'] == layer) &
                (df['checkpoint_str'] == checkpoint)
            ]
            
            if not filtered_df.empty:
                row = filtered_df.iloc[0]
                
                if 'variance_explained' in row:
                    var_expl_str = row['variance_explained']
                    debug_info.append(f"variance_explained type: {type(var_expl_str).__name__}")
                    debug_info.append(f"variance_explained value: {repr(var_expl_str)[:200]}")
                    
                    # Add a more complete view of the variance_explained value
                    if isinstance(var_expl_str, str) and len(var_expl_str) > 200:
                        debug_info.append(f"Full variance_explained string length: {len(var_expl_str)}")
                        debug_info.append(f"Last 50 characters: {repr(var_expl_str[-50:] if len(var_expl_str) > 50 else var_expl_str)}")
                        # Check if string is incomplete (missing closing bracket)
                        if var_expl_str.count('[') > var_expl_str.count(']'):
                            debug_info.append("WARNING: Unbalanced brackets detected!")
                            # Try to fix the string by adding missing brackets
                            fixed_str = var_expl_str
                            if var_expl_str.startswith('[') and not var_expl_str.endswith(']'):
                                fixed_str = var_expl_str + ']'
                                debug_info.append(f"Added closing bracket. New length: {len(fixed_str)}")
                                
                                # Try parsing with fixed string
                                debug_info.append("\nTrying parsing with fixed string:")
                                try:
                                    import ast
                                    eval_result = ast.literal_eval(fixed_str)
                                    debug_info.append(f"ast.literal_eval successful with fixed string: {type(eval_result).__name__}, length: {len(eval_result)}")
                                    debug_info.append(f"First 5 values: {eval_result[:5]}")
                                    debug_info.append(f"Last 5 values: {eval_result[-5:]}")
                                except Exception as e:
                                    debug_info.append(f"ast.literal_eval still failed with fixed string: {str(e)}")
                    
                    # Try different parsing approaches
                    debug_info.append("\nTrying different parsing approaches:")
                    
                    # 1. Direct use if already a list
                    debug_info.append(f"Is already a list? {isinstance(var_expl_str, list)}")
                    
                    # 2. JSON parsing
                    try:
                        if isinstance(var_expl_str, str):
                            var_expl_str_cleaned = var_expl_str.strip()
                            debug_info.append(f"Starts with '[' and ends with ']'? {var_expl_str_cleaned.startswith('[') and var_expl_str_cleaned.endswith(']')}")
                            json_result = json.loads(var_expl_str_cleaned)
                            debug_info.append(f"JSON parsing successful: {type(json_result).__name__}, length: {len(json_result)}")
                        else:
                            debug_info.append("Skipping JSON parsing (not a string)")
                    except Exception as e:
                        debug_info.append(f"JSON parsing failed: {str(e)}")
                    
                    # 3. ast.literal_eval
                    try:
                        if isinstance(var_expl_str, str):
                            import ast
                            eval_result = ast.literal_eval(var_expl_str)
                            debug_info.append(f"ast.literal_eval successful: {type(eval_result).__name__}, length: {len(eval_result)}")
                        else:
                            debug_info.append("Skipping ast.literal_eval (not a string)")
                    except Exception as e:
                        debug_info.append(f"ast.literal_eval failed: {str(e)}")
                    
                    # 4. Try extracting values directly
                    try:
                        if hasattr(var_expl_str, '__iter__') and not isinstance(var_expl_str, str):
                            list_result = list(var_expl_str)
                            debug_info.append(f"Direct list conversion successful: length: {len(list_result)}")
                            if list_result:
                                debug_info.append(f"First few elements: {list_result[:5]}")
                        else:
                            debug_info.append("Skipping direct list conversion (not iterable or is string)")
                    except Exception as e:
                        debug_info.append(f"Direct list conversion failed: {str(e)}")
                    
                else:
                    debug_info.append("No 'variance_explained' column found in the DataFrame.")
            else:
                debug_info.append(f"No data found with target='{target}', layer='{layer}', checkpoint='{checkpoint}'")
        except Exception as e:
            debug_info.append(f"Error analyzing variance_explained: {str(e)}")
            import traceback
            debug_info.append(traceback.format_exc())
    else:
        debug_info.append("Select target, layer, and checkpoint to analyze variance_explained data.")
    
    return html.Pre("\n".join(debug_info))

# Add callback for layer-target-heatmap
@app.callback(
    Output('layer-target-heatmap', 'figure'),
    Input('update-plots-button', 'n_clicks'),
    State('current-df', 'data'),
    State('checkpoint-dropdown', 'value'),
    State('rcond-input', 'value'),
    State('metric-dropdown', 'value'),
    State('normalization-dropdown', 'value'),
    prevent_initial_call=True
)
def update_layer_vs_target(n_clicks, df_json, checkpoint, rcond, metric, normalization='none'):
    """
    Update the Layer vs Target heatmap based on selected checkpoint and filters.
    Creates a heatmap where layers are rows and targets are columns.
    """
    if df_json is None:
        return go.Figure().update_layout(
            title="No data loaded",
            height=400
        )
    
    try:
        # Parse the JSON data
        df = pd.read_json(StringIO(df_json), orient='split')
        if df.empty:
            return go.Figure().update_layout(
                title="No data available",
                height=400
            )
        
        # Check if metric is valid
        metric_info = METRICS.get(metric, {'title': metric, 'description': ''})
        
        # Filter by checkpoint and convert to float for comparison
        checkpoint_filter = df['checkpoint'] == float(checkpoint)
        if not any(checkpoint_filter):
            return go.Figure().update_layout(
                title=f"No data for checkpoint {checkpoint}",
                height=400
            )
            
        # Apply normalization if needed
        if normalization != 'none':
            df = normalize_data(df, normalization)
        
        # Determine if we use the best rcond or a specific value
        using_best_rcond = rcond.lower() == 'best'
        
        if using_best_rcond:
            # Use the best rcond for each layer and target
            targets = df['target'].unique()
            layers = df['layer_name'].unique()
            best_rows = []
            
            for target in targets:
                # Find best rcond for each layer
                better = 'min' if metric == 'norm_dist' else 'max'
                best_rconds = find_best_rconds_for_all_layers(df, target, checkpoint, layers, metric, better)
                
                # Add the best rows to our list
                for layer, best_rcond in best_rconds.items():
                    # Get the row with the best rcond
                    best_row = df[
                        (df['target'] == target) &
                        (df['checkpoint'] == float(checkpoint)) &
                        (df['layer_name'] == layer) &
                        np.isclose(df['rcond'], best_rcond, rtol=1e-5, atol=1e-12)
                    ]
                    
                    if not best_row.empty:
                        best_rows.append(best_row)
            
            # Combine all best rows
            if best_rows:
                filtered_df = pd.concat(best_rows, ignore_index=True)
            else:
                filtered_df = pd.DataFrame()
        else:
            # Filter with specific rcond value
            try:
                rcond_filter = np.isclose(df['rcond'], float(rcond), rtol=1e-5, atol=1e-12)
                filtered_df = df[
                    (df['checkpoint'] == float(checkpoint)) & 
                    rcond_filter
                ]
            except ValueError:
                # Handle invalid rcond
                return go.Figure().update_layout(
                    title=f"Invalid rcond value: {rcond}",
                    height=400
                )
        
        if filtered_df.empty:
            rcond_display = "best" if using_best_rcond else f"{float(rcond):.2e}"
            msg = f"No data found with checkpoint='{checkpoint}', rcond={rcond_display}"
            return go.Figure().update_layout(
                title=msg,
                height=400
            )
        
        # Create a pivot table
        pivot_df = filtered_df.pivot_table(
            index='layer_name', 
            columns='target', 
            values=metric,
            aggfunc='mean'
        )
        
        # Choose color scale based on metric (reverse for norm_dist)
        color_scale = "Viridis_r" if metric == 'norm_dist' else "Viridis"
        
        # Create a heatmap
        rcond_display = "best" if using_best_rcond else f"{float(rcond):.2e}"
        fig = px.imshow(
            pivot_df,
            labels=dict(x="Target", y="Layer", color=metric_info['title']),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale=color_scale,
            title=f"Layer vs Target Performance (Checkpoint {checkpoint}, rcond={rcond_display})",
            height=400
        )
        
        # Improve layout
        fig.update_layout(
            margin=dict(t=60, r=50, l=150),  # More space for layer names
            coloraxis_colorbar=dict(
                title=f"{metric_info['title']}<br>({metric_info['description']})"
            )
        )
        
        return fig
    
    except Exception as e:
        error_msg = f"Error generating layer vs target heatmap: {str(e)}"
        return go.Figure().update_layout(
            title=error_msg,
            height=400
        )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)