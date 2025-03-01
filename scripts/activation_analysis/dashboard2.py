"""
Enhanced Dashboard for Activation Analysis

This dashboard visualizes activation analysis results for neural networks, working with the unified h5 file format.
It includes multiple visualization options including checkpoint evolution, layer performance,
random baseline comparison, and singular value analysis.
"""
import dash  # type: ignore
from dash import dcc, html, Input, Output, State, callback  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import pandas as pd
import numpy as np
import h5py  # type: ignore
import os
import glob
import json
import logging
from io import StringIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define metrics with display information
METRICS = {
    'norm_dist': {
        'title': 'Normalized Distance',
        'description': 'Lower is better',
        'better': 'min',
        'color_scale': 'Viridis'
    },
    'r_squared': {
        'title': 'R²',
        'description': 'Higher is better',
        'better': 'max',
        'color_scale': 'Viridis_r'
    },
    'variance_explained': {
        'title': 'Variance Explained',
        'description': 'Higher is better',
        'better': 'max',
        'color_scale': 'Viridis_r'
    }
}

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Activation Analysis Dashboard v2"

# Define the layout with tabs for different visualizations
app.layout = html.Div([
    html.H1("Activation Analysis Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Controls Panel
    html.Div([
        html.Div([
            html.H4("Data Selection"),
            html.Label("Data Directory:"),
            dcc.Input(
                id="data-directory",
                type="text",
                value="./analysis",
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            html.Button('Load Runs', id='load-runs-button', n_clicks=0, style={'marginBottom': '10px'}),
            
            html.Label("Run:"),
            dcc.Dropdown(id='run-id-dropdown', placeholder="Select a run"),
            
            html.Label("Target:"),
            dcc.Dropdown(id='target-dropdown'),
            
            html.Label("Metric:"),
            dcc.RadioItems(
                id='metric-selector',
                options=[
                    {'label': 'Normalized Distance (lower is better)', 'value': 'norm_dist'},
                    {'label': 'R² (higher is better)', 'value': 'r_squared'}
                ],
                value='norm_dist',
                labelStyle={'display': 'block', 'marginBottom': '5px'}
            ),
            
            html.Hr(),
            
            html.Label("Layer:"),
            dcc.Dropdown(id='layer-dropdown'),
            
            html.Label("Checkpoint:"),
            dcc.Dropdown(id='checkpoint-dropdown', multi=True),
            
            html.Label("Regularization Parameter (rcond):"),
            dcc.Dropdown(id='rcond-dropdown'),
            
            html.Hr(),
            
            html.Button('Update Plots', id='update-button', n_clicks=0),
            
            html.Div(id='dataset-info', style={'marginTop': '20px', 'fontSize': '12px'})
        ], className="three columns", style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Main content area with tabs
        html.Div([
            dcc.Tabs([
                dcc.Tab(label="Layer Performance", children=[
                    html.Div([
                        html.H4("Layer Performance Comparison", style={'textAlign': 'center'}),
                        dcc.Graph(id='layer-performance-chart'),
                    ], className="row"),
                ]),
                
                dcc.Tab(label="Checkpoint Evolution", children=[
                    html.Div([
                        html.H4("Training Dynamics", style={'textAlign': 'center'}),
                        dcc.Graph(id='checkpoint-evolution'),
                    ], className="row"),
                ]),
                
                dcc.Tab(label="Random Baseline", children=[
                    html.Div([
                        html.H4("Comparison to Random Baseline", style={'textAlign': 'center'}),
                        dcc.Graph(id='random-baseline-comparison'),
                    ], className="row"),
                ]),
                
                dcc.Tab(label="Layer vs Target", children=[
                    html.Div([
                        html.H4("Layer vs Target Heatmap", style={'textAlign': 'center'}),
                        dcc.Graph(id='layer-target-heatmap'),
                        html.Div([
                            html.Button('Update Heatmap', id='update-heatmap-button', n_clicks=0, style={'marginTop': '10px'})
                        ], style={'margin': '10px', 'textAlign': 'center'}),
                    ], className="row"),
                ]),
                
                dcc.Tab(label="Singular Values", children=[
                    html.Div([
                        html.H4("Singular Value Distribution", style={'textAlign': 'center'}),
                        dcc.Graph(id='singular-value-plot'),
                    ], className="row"),
                ]),
            ]),
        ], className="nine columns"),
    ], className="row"),
    
    # Store components for intermediate data
    dcc.Store(id='h5-data'),
    dcc.Store(id='current-df'),
    dcc.Store(id='singular-values-data'),
    dcc.Store(id='available-layers'),
    dcc.Store(id='available-checkpoints'),
    dcc.Store(id='available-rconds'),
])

#
# Helper Functions
#

def find_h5_files(directory):
    """Find all H5 files in the directory that match the unified format pattern."""
    if not os.path.exists(directory):
        return []
    
    # Get all H5 files with unified format pattern
    pattern = os.path.join(directory, "**", "*_unified_results.h5")
    files = glob.glob(pattern, recursive=True)
    
    logger.info(f"Found {len(files)} potential H5 files in {directory}")
    return files

def load_unified_h5_data(h5_file):
    """Load data from a unified H5 file format."""
    try:
        result = {
            'checkpoint_df': None,
            'random_df': None,
            'singular_values': {},
            'weights': {},
            'attrs': {}
        }
        
        with h5py.File(h5_file, 'r') as f:
            # Check if this is a unified format file
            if 'unified_format' not in f.attrs or not f.attrs['unified_format']:
                logger.warning(f"File {h5_file} is not in unified format")
                return None
            
            # Get global attributes
            for key in f.attrs:
                attr_value = f.attrs[key]
                if isinstance(attr_value, bytes):
                    attr_value = attr_value.decode('utf-8')
                result['attrs'][key] = attr_value
            
            # Load checkpoint results
            if 'checkpoint_results' in f:
                checkpoint_group = f['checkpoint_results']
                
                # Load regression results
                if 'regression_results' in checkpoint_group:
                    df_dict = {}
                    for col in checkpoint_group['regression_results']:
                        data = checkpoint_group['regression_results'][col][()]
                        if data.dtype.kind == 'S':  # Handle byte strings
                            data = np.array([s.decode('utf-8') for s in data])
                        df_dict[col] = data
                    
                    df = pd.DataFrame(df_dict)
                    df['is_random'] = False
                    
                    # Add back metadata as columns if not already present
                    if 'run_id' not in df.columns and 'run_id' in result['attrs']:
                        df['run_id'] = result['attrs']['run_id']
                    if 'sweep_id' not in df.columns and 'sweep_id' in result['attrs']:
                        df['sweep_id'] = result['attrs']['sweep_id']
                    
                    result['checkpoint_df'] = df
                
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
                                
                                entry_data = {
                                    "singular_values": entry['singular_values'][()]
                                }
                                
                                if "checkpoint" in entry.attrs:
                                    entry_data['checkpoint'] = str(entry.attrs['checkpoint'])
                                
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
                    
                    df = pd.DataFrame(df_dict)
                    df['is_random'] = True
                    
                    # Add back metadata as columns if not already present
                    if 'run_id' not in df.columns and 'run_id' in result['attrs']:
                        df['run_id'] = result['attrs']['run_id']
                    if 'sweep_id' not in df.columns and 'sweep_id' in result['attrs']:
                        df['sweep_id'] = result['attrs']['sweep_id']
                    
                    result['random_df'] = df
        
        return result
    except Exception as e:
        logger.error(f"Error loading unified H5 data: {e}")
        return None

def combine_checkpoint_and_random_data(checkpoint_df, random_df):
    """Combine checkpoint and random baseline data into a single DataFrame."""
    if checkpoint_df is None and random_df is None:
        return None
    
    if checkpoint_df is None:
        return random_df
    
    if random_df is None:
        return checkpoint_df
    
    # Ensure both DataFrames have the 'is_random' column
    if 'is_random' not in checkpoint_df.columns:
        checkpoint_df['is_random'] = False
    if 'is_random' not in random_df.columns:
        random_df['is_random'] = True
    
    # Combine the DataFrames
    combined_df = pd.concat([checkpoint_df, random_df], ignore_index=True)
    return combined_df

def find_best_rcond(df, target, checkpoint, layer, metric='norm_dist', better='min'):
    """Find the best rcond value for a given target/checkpoint/layer combination."""
    # Ensure checkpoint is a string for comparison
    checkpoint_str = str(checkpoint)
    
    # Filter DataFrame
    filtered_df = df[
        (df['target'] == target) &
        (df['checkpoint'].astype(str) == checkpoint_str) &
        (df['layer_name'] == layer)
    ]
    
    if filtered_df.empty:
        return None
    
    # Find best rcond based on the metric
    if better == 'min':  # Lower is better (e.g., norm_dist)
        idx = filtered_df[metric].idxmin()
    else:  # Higher is better (e.g., r_squared)
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

def numpy_to_python_types(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python_types(i) for i in obj]
    elif isinstance(obj, np.number):
        return obj.item()
    else:
        return obj

def extract_checkpoint_number(checkpoint_name):
    """Extract numeric checkpoint value for sorting."""
    try:
        if isinstance(checkpoint_name, (int, float)):
            return float(checkpoint_name)
        
        if 'RANDOM' in str(checkpoint_name):
            return -1  # Place random at the beginning
        
        # Try to extract numbers from string
        checkpoint_str = str(checkpoint_name)
        if checkpoint_str.isdigit():
            return int(checkpoint_str)
        
        # Look for '_' separator pattern
        if '_' in checkpoint_str:
            parts = checkpoint_str.split('_')
            for part in parts:
                if part.isdigit():
                    return int(part)
        
        # As a fallback, try to find any digits in the string
        import re
        numbers = re.findall(r'\d+', checkpoint_str)
        if numbers:
            return int(numbers[0])
        
        return 0  # Default value if no number found
    except:
        return 0

def json_to_dataframe(df_json):
    """Safely convert JSON string to pandas DataFrame."""
    if not df_json:
        return None
    try:
        return pd.read_json(StringIO(df_json), orient='split')
    except:
        logger.error("Error converting JSON to DataFrame")
        return None

#
# Callbacks
#

@app.callback(
    [Output('run-id-dropdown', 'options'),
     Output('dataset-info', 'children')],
    [Input('load-runs-button', 'n_clicks')],
    [State('data-directory', 'value')]
)
def load_available_runs(n_clicks, directory):
    """Load available runs from the specified directory."""
    if n_clicks == 0:
        return [], html.Div("Enter a data directory and click 'Load Runs'")
    
    if not directory or not os.path.exists(directory):
        return [], html.Div(f"Directory not found: {directory}", style={'color': 'red'})
    
    # Find unified H5 files
    h5_files = find_h5_files(directory)
    
    if not h5_files:
        return [], html.Div(f"No unified H5 files found in {directory}", style={'color': 'orange'})
    
    # Extract run options
    run_options = []
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'unified_format' in f.attrs and f.attrs['unified_format']:
                    # Get run_id and sweep_id
                    run_id = f.attrs.get('run_id', os.path.basename(h5_file).split('_unified_results.h5')[0])
                    if isinstance(run_id, bytes):
                        run_id = run_id.decode('utf-8')
                    
                    sweep_id = f.attrs.get('sweep_id', 'Unknown')
                    if isinstance(sweep_id, bytes):
                        sweep_id = sweep_id.decode('utf-8')
                    
                    run_options.append({
                        'label': f"{run_id} (Sweep: {sweep_id})",
                        'value': h5_file  # Store the full path
                    })
        except Exception as e:
            logger.error(f"Error reading H5 file {h5_file}: {e}")
    
    if not run_options:
        return [], html.Div(f"No valid unified H5 files found in {directory}", style={'color': 'red'})
    
    # Success message
    dataset_info = html.Div([
        html.P(f"Found {len(run_options)} valid H5 files in {directory}"),
        html.P("Select a run to analyze", style={'fontWeight': 'bold'})
    ])
    
    return run_options, dataset_info

@app.callback(
    [Output('h5-data', 'data'),
     Output('current-df', 'data'),
     Output('singular-values-data', 'data'),
     Output('target-dropdown', 'options'),
     Output('layer-dropdown', 'options'),
     Output('available-layers', 'data'),
     Output('checkpoint-dropdown', 'options'),
     Output('available-checkpoints', 'data'),
     Output('rcond-dropdown', 'options'),
     Output('available-rconds', 'data')],
    [Input('run-id-dropdown', 'value')]
)
def load_run_data(h5_file):
    # Initialize empty values
    current_df = None
    singular_values = None
    target_options = []
    layer_options = []
    available_layers = []
    checkpoint_options = []
    available_checkpoints = []
    rcond_options = []
    available_rconds = []
    
    # Return empty data if no run is selected
    if not h5_file:
        return json.dumps({"path": None}), None, None, [], [], [], [], [], [], []
    
    try:
        # Convert h5_file to string if it's bytes
        if isinstance(h5_file, bytes):
            h5_file = h5_file.decode('utf-8')
            
        # Load the H5 file and extract data
        unified_data = load_unified_h5_data(h5_file)
        
        if unified_data:
            # Convert unified_data to DataFrame and combine checkpoint + random data
            debug_info = []
            debug_info.append(f"Loading unified H5 data from {h5_file}")
            
            # Handle both checkpoint and random data if available
            combined_df = None
            if 'checkpoint_df' in unified_data and unified_data['checkpoint_df'] is not None:
                if 'random_df' in unified_data and unified_data['random_df'] is not None:
                    # Combine checkpoint and random data
                    combined_df = combine_checkpoint_and_random_data(
                        unified_data['checkpoint_df'], 
                        unified_data['random_df']
                    )
                    if combined_df is not None:
                        debug_info.append(f"Combined checkpoint and random data: {len(combined_df)} rows")
                else:
                    # Just use checkpoint data
                    combined_df = unified_data['checkpoint_df']
                    if combined_df is not None:
                        debug_info.append(f"Using checkpoint data only: {len(combined_df)} rows")
            
            # Set current dataframe
            current_df = combined_df
            
            # Extract singular values and weights
            singular_values = unified_data.get('singular_values', {})
            
            if current_df is not None:
                # Extract unique targets, layers, and checkpoints
                unique_targets = current_df['target'].unique()
                # Decode byte strings to regular strings if needed
                decoded_targets = []
                for target in unique_targets:
                    if isinstance(target, bytes):
                        decoded_targets.append(target.decode('utf-8'))
                    else:
                        decoded_targets.append(target)
                unique_targets = decoded_targets
                
                # Sort targets with nn_beliefs first if present
                if 'nn_beliefs' in unique_targets:
                    unique_targets = ['nn_beliefs'] + [t for t in sorted(unique_targets) if t != 'nn_beliefs']
                else:
                    unique_targets = sorted(unique_targets)
                
                target_options = [{'label': t, 'value': t} for t in unique_targets]
                
                # Extract all layers
                unique_layers = current_df['layer_name'].unique()
                # Decode byte strings to regular strings if needed
                decoded_layers = []
                for layer in unique_layers:
                    if isinstance(layer, bytes):
                        decoded_layers.append(layer.decode('utf-8'))
                    else:
                        decoded_layers.append(layer)
                unique_layers = decoded_layers
                
                # Sort layers alphabetically, but ensure 'model_final' is last if present
                if 'model_final' in unique_layers:
                    unique_layers = [l for l in sorted(unique_layers) if l != 'model_final'] + ['model_final']
                else:
                    unique_layers = sorted(unique_layers)
                
                layer_options = [{'label': l, 'value': l} for l in unique_layers]
                available_layers = unique_layers
                
                # Extract unique checkpoints and sort by numerical value
                unique_checkpoints = current_df['checkpoint'].unique()
                # Decode byte strings to regular strings if needed
                decoded_checkpoints = []
                for ckpt in unique_checkpoints:
                    if isinstance(ckpt, bytes):
                        decoded_checkpoints.append(ckpt.decode('utf-8'))
                    else:
                        decoded_checkpoints.append(ckpt)
                unique_checkpoints = decoded_checkpoints
                
                # Handle checkpoint sorting
                try:
                    # Try to sort checkpoints as integers
                    sorted_checkpoints = sorted(
                        [c for c in unique_checkpoints if not str(c).startswith('RANDOM')],
                        key=lambda x: extract_checkpoint_number(x)
                    )
                    
                    # Add random checkpoints at the end
                    random_checkpoints = sorted([c for c in unique_checkpoints if str(c).startswith('RANDOM')])
                    sorted_checkpoints.extend(random_checkpoints)
                except:
                    # If sorting as integers fails, sort as strings
                    sorted_checkpoints = sorted(unique_checkpoints)
                
                checkpoint_options = [{'label': c, 'value': c} for c in sorted_checkpoints]
                available_checkpoints = sorted_checkpoints
                
                # Extract unique rcond values and sort
                unique_rconds = sorted(current_df['rcond'].unique())
                rcond_options = [{'label': f'{r:.2e}', 'value': f'{r:.2e}'} for r in unique_rconds]
                rcond_options.append({'label': 'Best per layer', 'value': 'best'})
                available_rconds = [f'{r:.2e}' for r in unique_rconds]
                
                debug_info.append(f"Extracted options: {len(target_options)} targets, {len(layer_options)} layers, "
                                 f"{len(checkpoint_options)} checkpoints, {len(rcond_options)} rcond values")
    
    except Exception as e:
        import traceback
        print(f"Error loading data: {str(e)}")
        print(traceback.format_exc())
        # Return empty values if anything fails
        return json.dumps({"path": str(h5_file)}), None, None, [], [], [], [], [], [], []
    
    # Return all the data and options
    return (
        json.dumps({"path": h5_file}),
        current_df.to_json(date_format='iso', orient='split') if current_df is not None else None,
        json.dumps(numpy_to_python_types(singular_values)) if singular_values else None,
        target_options,
        layer_options,
        json.dumps(available_layers),
        checkpoint_options,
        json.dumps(available_checkpoints),
        rcond_options,
        json.dumps(available_rconds)
    )

@app.callback(
    Output('layer-performance-chart', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('current-df', 'data'),
     State('target-dropdown', 'value'),
     State('checkpoint-dropdown', 'value'),
     State('rcond-dropdown', 'value'),
     State('metric-selector', 'value'),
     State('available-layers', 'data')]
)
def update_layer_performance(n_clicks, df_json, target, checkpoint, rcond, metric, layers_json):
    """Update the layer performance chart showing metrics for each layer."""
    if not df_json or not target or not checkpoint:
        return go.Figure().update_layout(title="No data or selections available")
    
    try:
        df = json_to_dataframe(df_json)
        if df is None or df.empty:
            return go.Figure().update_layout(title="No data available")
        
        # Get metric info
        metric_info = METRICS.get(metric, {'title': metric, 'description': '', 'better': 'min'})
        better = metric_info['better']  # min or max
        
        # Handle checkpoint selection (use first if multiple selected)
        if isinstance(checkpoint, list):
            if not checkpoint:  # Empty list
                return go.Figure().update_layout(title="No checkpoint selected")
            checkpoint = checkpoint[0]  # Take the first one
        
        # Filter for target and checkpoint
        checkpoint_filter = df['checkpoint'].astype(str) == str(checkpoint)
        target_filter = df['target'] == target
        base_filter = checkpoint_filter & target_filter & (df['is_random'] == False)
        
        # First ensure we have data for this checkpoint/target
        if not any(base_filter):
            return go.Figure().update_layout(
                title=f"No data found for target='{target}', checkpoint='{checkpoint}'")
        
        # Handle 'best' rcond or specific value
        using_best_rcond = rcond == 'best'
        if using_best_rcond:
            # Get available layers
            available_layers = json.loads(layers_json) if layers_json else []
            
            # Find best rcond for each layer based on the metric
            best_rows = []
            for layer in available_layers:
                best_rcond_val = find_best_rcond(df, target, checkpoint, layer, metric, better)
                if best_rcond_val is not None:
                    layer_filter = (df['layer_name'] == layer) & \
                                    np.isclose(df['rcond'], best_rcond_val)
                    layer_row = df[base_filter & layer_filter]
                    if not layer_row.empty:
                        best_rows.append(layer_row)
            
            if not best_rows:
                return go.Figure().update_layout(
                    title=f"No best rcond data found for target='{target}', checkpoint='{checkpoint}'")
            
            # Combine rows and sort by layer
            filtered_df = pd.concat(best_rows).sort_values('layer_name')
        else:
            # Use specific rcond value
            try:
                rcond_val = float(rcond)
                rcond_filter = np.isclose(df['rcond'], rcond_val)
                filtered_df = df[base_filter & rcond_filter].sort_values('layer_name')
                
                if filtered_df.empty:
                    return go.Figure().update_layout(
                        title=f"No data found with rcond={rcond_val:.2e} for target='{target}', checkpoint='{checkpoint}'")
            except ValueError:
                return go.Figure().update_layout(
                    title=f"Invalid rcond value: {rcond}")
        
        # Create bar chart
        rcond_display = "best" if using_best_rcond else f"{float(rcond):.2e}"
        fig = px.bar(
            filtered_df,
            x='layer_name',
            y=metric,
            title=f"Layer Performance: {target} (Checkpoint {checkpoint}, rcond={rcond_display})",
            labels={'layer_name': 'Layer', metric: metric_info['title']},
            color='layer_name'
        )
        
        # Improve layout
        fig.update_layout(
            xaxis_title="Layer",
            yaxis_title=f"{metric_info['title']} ({metric_info['description']})",
            showlegend=False,
            height=500
        )
        
        # Add reference line for random baseline if available
        random_filter = (df['target'] == target) & (df['is_random'] == True)
        if any(random_filter):
            random_data = df[random_filter]
            # Calculate mean of random baseline
            random_mean = random_data[metric].mean()
            
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(filtered_df['layer_name'].unique()) - 0.5,
                y0=random_mean,
                y1=random_mean,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add annotation for random baseline
            fig.add_annotation(
                x=len(filtered_df['layer_name'].unique()) / 2,
                y=random_mean,
                text=f"Random Baseline: {random_mean:.4f}",
                showarrow=False,
                yshift=10,
                font=dict(color="red")
            )
        
        return fig
        
    except Exception as e:
        import traceback
        logger.error(f"Error in layer performance chart: {e}")
        logger.error(traceback.format_exc())
        return go.Figure().update_layout(
            title="Error updating chart",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'x': 0.5,
                'y': 0.5
            }]
        )

@app.callback(
    Output('checkpoint-evolution', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('current-df', 'data'),
     State('target-dropdown', 'value'),
     State('layer-dropdown', 'value'),
     State('rcond-dropdown', 'value'),
     State('metric-selector', 'value')]
)
def update_checkpoint_evolution(n_clicks, df_json, target, layer, rcond, metric):
    """Update the checkpoint evolution chart showing how metrics change during training."""
    if not df_json or not target or not layer:
        return go.Figure().update_layout(title="Please select target and layer")
    
    try:
        df = json_to_dataframe(df_json)
        if df is None or df.empty:
            return go.Figure().update_layout(title="No data available")
        
        # Get metric info
        metric_info = METRICS.get(metric, {'title': metric, 'description': '', 'better': 'min'})
        better = metric_info['better']  # min or max
        
        # Filter for the target and layer
        target_filter = df['target'] == target
        layer_filter = df['layer_name'] == layer
        non_random_filter = df['is_random'] == False
        
        # Combined filter
        base_filter = target_filter & layer_filter & non_random_filter
        
        if not any(base_filter):
            return go.Figure().update_layout(
                title=f"No data found for target='{target}', layer='{layer}'")
        
        # Handle 'best' rcond or specific value
        using_best_rcond = rcond == 'best'
        
        if using_best_rcond:
            # For each checkpoint, find the best rcond
            checkpoints = df[base_filter]['checkpoint'].unique()
            best_rows = []
            
            for ckpt in checkpoints:
                best_rcond_val = find_best_rcond(df, target, ckpt, layer, metric, better)
                if best_rcond_val is not None:
                    checkpoint_filter = df['checkpoint'] == ckpt
                    rcond_filter = np.isclose(df['rcond'], best_rcond_val)
                    best_row = df[base_filter & checkpoint_filter & rcond_filter]
                    
                    if not best_row.empty:
                        best_rows.append(best_row)
            
            if not best_rows:
                return go.Figure().update_layout(
                    title=f"No best rcond data found for target='{target}', layer='{layer}'")
            
            filtered_df = pd.concat(best_rows, ignore_index=True)
        else:
            # Filter for a specific rcond
            try:
                rcond_val = float(rcond)
                rcond_filter = np.isclose(df['rcond'], rcond_val)
                filtered_df = df[base_filter & rcond_filter]
                
                if filtered_df.empty:
                    return go.Figure().update_layout(
                        title=f"No data found with rcond={rcond_val:.2e} for target='{target}', layer='{layer}'")
            except ValueError:
                return go.Figure().update_layout(
                    title=f"Invalid rcond value: {rcond}")
        
        # Numerical conversion for sorting
        filtered_df['checkpoint_num'] = filtered_df['checkpoint'].apply(extract_checkpoint_number)
        filtered_df = filtered_df.sort_values('checkpoint_num')
        
        # Create line chart for the evolution
        rcond_display = "best" if using_best_rcond else f"{float(rcond):.2e}"
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=filtered_df['checkpoint_num'],
            y=filtered_df[metric],
            mode='lines+markers',
            name=f'{layer} ({target})',
            line=dict(width=2),
            marker=dict(size=8)
        ))
        
        # Add random baseline if available
        random_filter = target_filter & layer_filter & (df['is_random'] == True)
        if any(random_filter):
            random_data = df[random_filter]
            random_mean = random_data[metric].mean()
            random_std = random_data[metric].std()
            
            # Add a horizontal line for random baseline mean
            fig.add_shape(
                type="line",
                x0=min(filtered_df['checkpoint_num']),
                x1=max(filtered_df['checkpoint_num']),
                y0=random_mean,
                y1=random_mean,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add shaded area for standard deviation
            fig.add_trace(go.Scatter(
                x=list(filtered_df['checkpoint_num']) + list(filtered_df['checkpoint_num'])[::-1],
                y=[random_mean + random_std] * len(filtered_df) + [random_mean - random_std] * len(filtered_df),
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='rgba(255, 0, 0, 0)'),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Add annotation for random baseline
            fig.add_annotation(
                x=min(filtered_df['checkpoint_num']),
                y=random_mean,
                text=f"Random Baseline: {random_mean:.4f} ± {random_std:.4f}",
                showarrow=True,
                arrowhead=1,
                xshift=-5,
                font=dict(color="red")
            )
        
        # Update layout
        fig.update_layout(
            title=f"Training Dynamics: {target} - {layer} (rcond={rcond_display})",
            xaxis_title="Checkpoint Number",
            yaxis_title=f"{metric_info['title']} ({metric_info['description']})",
            height=500,
            hovermode="closest"
        )
        
        return fig
        
    except Exception as e:
        import traceback
        logger.error(f"Error in checkpoint evolution: {e}")
        logger.error(traceback.format_exc())
        return go.Figure().update_layout(
            title="Error updating chart",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'x': 0.5,
                'y': 0.5
            }]
        )

@app.callback(
    Output('random-baseline-comparison', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('current-df', 'data'),
     State('target-dropdown', 'value'),
     State('layer-dropdown', 'value'),
     State('rcond-dropdown', 'value'),
     State('metric-selector', 'value')]
)
def update_random_baseline_comparison(n_clicks, df_json, target, layer, rcond, metric):
    """Update the random baseline comparison chart."""
    if not df_json or not target or not layer:
        return go.Figure().update_layout(title="Please select target and layer")
    
    try:
        df = json_to_dataframe(df_json)
        if df is None or df.empty:
            return go.Figure().update_layout(title="No data available")
        
        # Check that we have random baseline data
        if 'is_random' not in df.columns:
            return go.Figure().update_layout(
                title="Random baseline information not available")
            
        random_data = df[df['is_random'] == True]
        trained_data = df[df['is_random'] == False]
        
        if random_data.empty:
            return go.Figure().update_layout(
                title="No random baseline data available")
        
        # Get metric info
        metric_info = METRICS.get(metric, {'title': metric, 'description': '', 'better': 'min'})
        
        # Filter for target and layer
        target_filter = df['target'] == target
        layer_filter = df['layer_name'] == layer
        
        random_filtered = random_data[target_filter & layer_filter]
        trained_filtered = trained_data[target_filter & layer_filter]
        
        if random_filtered.empty or trained_filtered.empty:
            return go.Figure().update_layout(
                title=f"No data for both trained and random with target='{target}', layer='{layer}'")
        
        # Handle 'best' rcond or specific value
        using_best_rcond = rcond == 'best'
        
        if using_best_rcond:
            # We'll need to find the best rcond for each checkpoint
            best_trained_rows = []
            checkpoints = trained_filtered['checkpoint'].unique()
            
            for ckpt in checkpoints:
                best_rcond_val = find_best_rcond(df, target, ckpt, layer, metric, metric_info['better'])
                if best_rcond_val is not None:
                    ckpt_filter = trained_filtered['checkpoint'] == ckpt
                    rcond_filter = np.isclose(trained_filtered['rcond'], best_rcond_val)
                    best_row = trained_filtered[ckpt_filter & rcond_filter]
                    
                    if not best_row.empty:
                        best_trained_rows.append(best_row)
            
            if not best_trained_rows:
                return go.Figure().update_layout(
                    title=f"No best rcond data for trained models with target='{target}', layer='{layer}'")
            
            trained_rows = pd.concat(best_trained_rows, ignore_index=True)
            
            # Same for random baseline
            best_random_rows = []
            random_ckpts = random_filtered['checkpoint'].unique()
            
            for ckpt in random_ckpts:
                best_rcond_val = find_best_rcond(df, target, ckpt, layer, metric, metric_info['better'])
                if best_rcond_val is not None:
                    ckpt_filter = random_filtered['checkpoint'] == ckpt
                    rcond_filter = np.isclose(random_filtered['rcond'], best_rcond_val)
                    best_row = random_filtered[ckpt_filter & rcond_filter]
                    
                    if not best_row.empty:
                        best_random_rows.append(best_row)
            
            if not best_random_rows:
                return go.Figure().update_layout(
                    title=f"No best rcond data for random baseline with target='{target}', layer='{layer}'")
            
            random_rows = pd.concat(best_random_rows, ignore_index=True)
        else:
            # Use specific rcond
            try:
                rcond_val = float(rcond)
                rcond_filter = np.isclose(df['rcond'], rcond_val)
                trained_rows = trained_filtered[rcond_filter]
                random_rows = random_filtered[rcond_filter]
                
                if trained_rows.empty or random_rows.empty:
                    return go.Figure().update_layout(
                        title=f"Insufficient data with rcond={rcond_val:.2e}")
            except ValueError:
                return go.Figure().update_layout(
                    title=f"Invalid rcond value: {rcond}")
        
        # Create violin plot comparison
        fig = go.Figure()
        
        # Add violin plots
        fig.add_trace(go.Violin(
            y=random_rows[metric],
            name='Random Baseline',
            box_visible=True,
            meanline_visible=True,
            marker_color='rgba(100, 100, 200, 0.5)',
            line_color='rgba(50, 50, 180, 1)',
        ))
        
        fig.add_trace(go.Violin(
            y=trained_rows[metric],
            name='Trained Models',
            box_visible=True,
            meanline_visible=True,
            marker_color='rgba(200, 100, 100, 0.5)',
            line_color='rgba(180, 50, 50, 1)',
        ))
        
        # Add mean lines
        random_mean = random_rows[metric].mean()
        trained_mean = trained_rows[metric].mean()
        
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
            text=f"Mean: {random_mean:.4f}",
            showarrow=True,
            arrowhead=1,
            yshift=10
        )
        
        fig.add_annotation(
            x=1,
            y=trained_mean,
            text=f"Mean: {trained_mean:.4f}",
            showarrow=True,
            arrowhead=1,
            yshift=10
        )
        
        # Update layout
        rcond_display = "best" if using_best_rcond else f"{float(rcond):.2e}"
        
        fig.update_layout(
            title=f"Random Baseline vs. Trained Models: {target} - {layer} (rcond={rcond_display})",
            xaxis_title="Model Type",
            yaxis_title=f"{metric_info['title']} ({metric_info['description']})",
            violinmode='group',
            height=500
        )
        
        return fig
        
    except Exception as e:
        import traceback
        logger.error(f"Error in random baseline comparison: {e}")
        logger.error(traceback.format_exc())
        return go.Figure().update_layout(
            title="Error updating chart",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'x': 0.5,
                'y': 0.5
            }]
        )

@app.callback(
    Output('layer-target-heatmap', 'figure'),
    [Input('update-heatmap-button', 'n_clicks')],
    [State('current-df', 'data'),
     State('checkpoint-dropdown', 'value'),
     State('rcond-dropdown', 'value'),
     State('metric-selector', 'value')]
)
def update_layer_target_heatmap(n_clicks, df_json, checkpoint, rcond, metric):
    """Update the layer vs target heatmap."""
    if not df_json or not checkpoint:
        return go.Figure().update_layout(title="Please select a checkpoint")
    
    try:
        df = json_to_dataframe(df_json)
        if df is None or df.empty:
            return go.Figure().update_layout(title="No data available")
        
        # Get metric info
        metric_info = METRICS.get(metric, {'title': metric, 'description': '', 'better': 'min', 'color_scale': 'Viridis'})
        better = metric_info['better']
        color_scale = metric_info['color_scale']
        
        # Handle checkpoint selection (use first if multiple)
        if isinstance(checkpoint, list):
            if not checkpoint:
                return go.Figure().update_layout(title="No checkpoint selected")
            checkpoint = checkpoint[0]  # Take the first one
        
        # Filter for non-random data and the selected checkpoint
        checkpoint_filter = df['checkpoint'].astype(str) == str(checkpoint)
        non_random_filter = df['is_random'] == False
        base_filter = checkpoint_filter & non_random_filter
        
        if not any(base_filter):
            return go.Figure().update_layout(
                title=f"No data found for checkpoint='{checkpoint}'")
        
        # Handle 'best' rcond or specific value
        using_best_rcond = rcond == 'best'
        
        if using_best_rcond:
            # We'll need to find the best rcond for each target/layer combination
            filtered_df = df[base_filter].copy()
            
            # Get unique targets and layers
            targets = filtered_df['target'].unique()
            layers = filtered_df['layer_name'].unique()
            
            # Create a pivot table to store the best values
            pivot_data = []
            
            for target in targets:
                for layer in layers:
                    target_layer_filter = (filtered_df['target'] == target) & (filtered_df['layer_name'] == layer)
                    if any(target_layer_filter):
                        # Find the best rcond
                        if better == 'min':
                            best_idx = filtered_df[target_layer_filter][metric].idxmin()
                        else:
                            best_idx = filtered_df[target_layer_filter][metric].idxmax()
                        
                        best_row = filtered_df.loc[best_idx]
                        pivot_data.append({
                            'target': target,
                            'layer_name': layer,
                            metric: best_row[metric]
                        })
            
            if not pivot_data:
                return go.Figure().update_layout(
                    title=f"No best rcond data found for checkpoint='{checkpoint}'")
            
            # Create DataFrame from pivot data
            pivot_df = pd.DataFrame(pivot_data)
        else:
            # Use specific rcond
            try:
                rcond_val = float(rcond)
                rcond_filter = np.isclose(df['rcond'], rcond_val)
                filtered_df = df[base_filter & rcond_filter]
                
                if filtered_df.empty:
                    return go.Figure().update_layout(
                        title=f"No data found with rcond={rcond_val:.2e} for checkpoint='{checkpoint}'")
                
                # Create a pivot table
                pivot_df = filtered_df.pivot_table(
                    index='layer_name', 
                    columns='target', 
                    values=metric,
                    aggfunc='mean'
                )
            except ValueError:
                return go.Figure().update_layout(
                    title=f"Invalid rcond value: {rcond}")
        
        # Create heatmap
        rcond_display = "best" if using_best_rcond else f"{float(rcond):.2e}"
        
        # Create the pivot table if we're using pivot_data
        if using_best_rcond:
            pivot_table = pivot_df.pivot_table(
                index='layer_name', 
                columns='target', 
                values=metric,
                aggfunc='mean'
            )
        else:
            pivot_table = pivot_df
        
        # Reverse color scale if metric is better when lower
        if better == 'min':
            if color_scale.endswith('_r'):
                color_scale = color_scale[:-2]
            else:
                color_scale = f"{color_scale}_r"
        
        fig = px.imshow(
            pivot_table,
            labels=dict(x="Target", y="Layer", color=metric_info['title']),
            x=pivot_table.columns,
            y=pivot_table.index,
            color_continuous_scale=color_scale,
            title=f"Layer vs Target Performance (Checkpoint {checkpoint}, rcond={rcond_display})",
            height=600
        )
        
        # Customize layout
        fig.update_layout(
            margin=dict(t=60, r=60, l=160),  # More space for layer names
            coloraxis_colorbar=dict(
                title=f"{metric_info['title']}<br>({metric_info['description']})"
            )
        )
        
        return fig
        
    except Exception as e:
        import traceback
        logger.error(f"Error in layer-target heatmap: {e}")
        logger.error(traceback.format_exc())
        return go.Figure().update_layout(
            title="Error updating chart",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'x': 0.5,
                'y': 0.5
            }]
        )

@app.callback(
    Output('singular-value-plot', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('singular-values-data', 'data'),
     State('target-dropdown', 'value'),
     State('layer-dropdown', 'value'),
     State('checkpoint-dropdown', 'value')]
)
def update_singular_value_plot(n_clicks, singular_values_json, target, layer, checkpoint):
    """Update the singular value distribution plot."""
    if not singular_values_json or not target or not layer:
        return go.Figure().update_layout(title="Please select target and layer")
    
    try:
        # Load singular values data
        singular_values = json.loads(singular_values_json)
        
        # Check if target and layer exist in the data
        if target not in singular_values or layer not in singular_values[target]:
            return go.Figure().update_layout(
                title=f"No singular values available for {layer} in {target}",
                height=500
            )
        
        # Get the singular values
        sv_data = singular_values[target][layer]
        
        # If checkpoint is selected, filter to that checkpoint
        if checkpoint:
            if isinstance(checkpoint, list):
                checkpoint_list = [str(c) for c in checkpoint]
                sv_data = [item for item in sv_data if 'checkpoint' in item and item['checkpoint'] in checkpoint_list]
            else:
                checkpoint_str = str(checkpoint)
                sv_data = [item for item in sv_data if 'checkpoint' in item and item['checkpoint'] == checkpoint_str]
        
        if not sv_data:
            return go.Figure().update_layout(
                title=f"No singular values available for selected checkpoint(s)",
                height=500
            )
        
        # Create figure with two subplots: linear and log scale
        fig = make_subplots(
            rows=2, 
            cols=1,
            subplot_titles=("Singular Values (Linear Scale)", "Singular Values (Log Scale)"),
            shared_xaxes=True
        )
        
        # Add a trace for each checkpoint
        for item in sv_data:
            sv = item['singular_values']
            
            # Get checkpoint for the label
            if 'checkpoint' in item:
                label = f"Checkpoint {item['checkpoint']}"
            elif 'random_idx' in item:
                label = f"Random {item['random_idx']}"
            else:
                label = "Unknown"
            
            # Add to linear scale subplot
            fig.add_trace(
                go.Scatter(
                    y=sv,
                    mode='lines',
                    name=label
                ),
                row=1, 
                col=1
            )
            
            # Add to log scale subplot
            fig.add_trace(
                go.Scatter(
                    y=sv,
                    mode='lines',
                    name=label,
                    showlegend=False
                ),
                row=2,
                col=1
            )
        
        # Update layout with titles and axis labels
        fig.update_layout(
            title=f"Singular Values for {layer} in {target}",
            height=700,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="closest"
        )
        
        # Update y-axis for log scale on the second subplot
        fig.update_yaxes(type="log", row=2, col=1)
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Component Index", row=2, col=1)
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Singular Value", row=1, col=1)
        fig.update_yaxes(title_text="Singular Value (Log Scale)", row=2, col=1)
        
        return fig
        
    except Exception as e:
        import traceback
        logger.error(f"Error in singular value plot: {e}")
        logger.error(traceback.format_exc())
        return go.Figure().update_layout(
            title="Error updating chart",
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'x': 0.5,
                'y': 0.5
            }]
        )

@app.callback(
    Output('target-dropdown', 'value'),
    [Input('run-id-dropdown', 'value')],
    [State('target-dropdown', 'options')]
)
def set_default_target(h5_file, target_options):
    """Set a default target when a run is selected."""
    if not h5_file or not target_options:
        return None
    
    # Return the first target option if available
    return target_options[0]['value'] if target_options else None

@app.callback(
    Output('checkpoint-dropdown', 'value'),
    [Input('run-id-dropdown', 'value')],
    [State('available-checkpoints', 'data')]
)
def set_default_checkpoint(h5_file, checkpoints_json):
    """Set a default checkpoint when a run is selected."""
    if not h5_file or not checkpoints_json:
        return None
    
    try:
        checkpoints = json.loads(checkpoints_json)
        if not checkpoints:
            return None
        
        # Return the last checkpoint (assuming this is the final model)
        return checkpoints[-1]
    except:
        return None

@app.callback(
    Output('layer-dropdown', 'value'),
    [Input('run-id-dropdown', 'value')],
    [State('layer-dropdown', 'options')]
)
def set_default_layer(h5_file, layer_options):
    """Set a default layer when a run is selected."""
    if not h5_file or not layer_options:
        return None
    
    # Return the first layer option if available
    return layer_options[0]['value'] if layer_options else None

@app.callback(
    Output('rcond-dropdown', 'value'),
    [Input('run-id-dropdown', 'value')],
    [State('rcond-dropdown', 'options')]
)
def set_default_rcond(h5_file, rcond_options):
    """Set default rcond value to 'best'."""
    if not h5_file or not rcond_options:
        return None
    
    # Default to 'best'
    return 'best'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
