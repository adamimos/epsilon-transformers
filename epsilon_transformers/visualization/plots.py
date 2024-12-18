import numpy as np
from jaxtyping import Float
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

def _project_to_simplex(points: Float[np.ndarray, "num_points num_states"]):
    """Project points onto the 2-simplex (equilateral triangle in 2D)."""
    x = points[:, 1] + 0.5 * points[:, 2]
    y = (np.sqrt(3) / 2) * points[:, 2]
    return x, y

def create_interactive_mse_plot_with_selector(loader, sweeps_dict):
    """
    Create an interactive Dash app with dataset selection capabilities.
    
    Args:
        loader: S3ModelLoader instance
        sweeps_dict (dict): Dictionary mapping sweep_ids to their descriptions
            e.g. {'20241121152808': 'RNN', '20241205175736': 'Transformer'}
    
    Returns:
        Dash: A configured Dash app ready to run
    """

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Create dataset selector components
    sweep_options = [{'label': f"{desc} ({sid})", 'value': sid} 
                    for sid, desc in sweeps_dict.items()]

    app.layout = html.Div([
        # Dataset Selection Section
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H4("Select Dataset"),
                        dcc.Dropdown(
                            id='sweep-dropdown',
                            options=sweep_options,
                            value=list(sweeps_dict.keys())[0],
                            className="mb-2"
                        ),
                        dcc.Dropdown(
                            id='run-dropdown',
                            placeholder="Select a run",
                            className="mb-2"
                        ),
                        dbc.Button("Load Dataset", id="load-button", color="primary")
                    ], width=6)
                ])
            ])
        ], className="mb-3"),

        # Main Plot Section (initially hidden)
        html.Div(id="plot-container", style={'display': 'none'}, children=[
            html.H1(id="plot-title", className="text-center mb-4"),
            
            dbc.Row([
                # Checkboxes column
                dbc.Col([
                    html.Div([
                        html.H4("Base Analysis Types"),
                        dcc.Checklist(id='base-analysis-checklist', className="mb-3"),
                        
                        html.H4("Orders"),
                        dcc.Checklist(id='order-checklist', className="mb-3"),
                        
                        html.H4("Metric Types"),
                        dcc.Checklist(id='metric-checklist', className="mb-3"),
                        
                        html.H4("Layers"),
                        dcc.Checklist(id='layer-checklist', className="mb-3"),
                        
                        html.H4("Scale"),
                        dbc.Switch(
                            id='log-scale-toggle',
                            label="Use Log Scale",
                            value=True,
                            className="mb-3"
                        ),

                        html.H4("Legend"),
                        dbc.Switch(
                            id='show-legend-toggle',
                            label="Show Legend",
                            value=True,
                            className="mb-3"
                        ),
                    ], className="p-3 border rounded")
                ], width=3),
                
                # Plot column
                dbc.Col([
                    dcc.Graph(id='analysis-plot', style={'height': '80vh'})
                ], width=9)
            ])
        ])
    ])

    # Callback to update run options based on sweep selection
    @app.callback(
        Output('run-dropdown', 'options'),
        Input('sweep-dropdown', 'value')
    )
    def update_run_options(sweep_id):
        if not sweep_id:
            return []
        runs = loader.list_runs_in_sweep(sweep_id)
        return [{'label': run_id, 'value': run_id} for run_id in runs]

    # Callback to load dataset and update plot options
    @app.callback(
        [Output('plot-container', 'style'),
         Output('plot-title', 'children'),
         Output('base-analysis-checklist', 'options'),
         Output('base-analysis-checklist', 'value'),
         Output('order-checklist', 'options'),
         Output('order-checklist', 'value'),
         Output('metric-checklist', 'options'),
         Output('metric-checklist', 'value'),
         Output('layer-checklist', 'options'),
         Output('layer-checklist', 'value')],
        Input('load-button', 'n_clicks'),
        [State('sweep-dropdown', 'value'),
         State('run-dropdown', 'value')],
        prevent_initial_call=True
    )
    def load_dataset(n_clicks, sweep_id, run_id):
        if not sweep_id or not run_id:
            return {'display': 'none'}, "", [], [], [], [], [], [], [], []

        # Load the dataset
        df = loader.load_mse_csv(sweep_id, run_id)
        
        # Prepare options and values for each checklist
        base_analysis_options = [{'label': x, 'value': x} for x in sorted(df['base_analysis'].unique())]
        order_options = [{'label': f"Order-{x}", 'value': x} for x in sorted(df['order'].unique())]
        metric_options = [{'label': x, 'value': x} for x in sorted(df['metric_type'].unique())]
        layer_options = [{'label': x, 'value': x} for x in sorted(df['layer'].unique())]

        return (
            {'display': 'block'},
            f"MSE Plot for {sweeps_dict[sweep_id]} - {run_id}",
            base_analysis_options,
            [opt['value'] for opt in base_analysis_options],
            order_options,
            [opt['value'] for opt in order_options],
            metric_options,
            [opt['value'] for opt in metric_options],
            layer_options,
            [opt['value'] for opt in layer_options]
        )

    # Callback for the plot updates
    @app.callback(
        Output('analysis-plot', 'figure'),
        [Input('base-analysis-checklist', 'value'),
         Input('order-checklist', 'value'),
         Input('metric-checklist', 'value'),
         Input('layer-checklist', 'value'),
         Input('log-scale-toggle', 'value'),
         Input('show-legend-toggle', 'value'),
         Input('sweep-dropdown', 'value'),
         Input('run-dropdown', 'value')]
    )
    def update_plot(selected_analyses, selected_orders, selected_metrics, selected_layers, 
                   use_log_scale, show_legend, sweep_id, run_id):
        if not all([sweep_id, run_id, selected_analyses, selected_orders, selected_metrics, selected_layers]):
            return go.Figure()

        # Load the dataset
        df = pd.read_csv(f"mse_df_{sweep_id}_{run_id}.csv")
        
        fig = go.Figure()
        
        # Filter data based on selections
        mask = (df['base_analysis'].isin(selected_analyses)) & \
               (df['order'].isin(selected_orders)) & \
               (df['metric_type'].isin(selected_metrics)) & \
               (df['layer'].isin(selected_layers))
        
        filtered_df = df[mask]
        
        for base_analysis in selected_analyses:
            for order in selected_orders:
                for metric in selected_metrics:
                    for layer in selected_layers:
                        sub_mask = (filtered_df['base_analysis'] == base_analysis) & \
                                  (filtered_df['order'] == order) & \
                                  (filtered_df['metric_type'] == metric) & \
                                  (filtered_df['layer'] == layer)
                        
                        if filtered_df[sub_mask].empty:
                            continue
                        
                        trace_name = f"{base_analysis} O{order} - {metric} - {layer}"
                        fig.add_trace(
                            go.Scatter(
                                x=filtered_df[sub_mask]['checkpoint'],
                                y=filtered_df[sub_mask]['value'],
                                name=trace_name,
                                mode='lines+markers',
                                hovertemplate=trace_name+'<extra></extra>'
                            )
                        )
        
        fig.update_layout(
            xaxis_title="Checkpoint",
            yaxis_title="Value",
            yaxis_type="log" if use_log_scale else "linear",
            template="plotly_white",
            showlegend=show_legend,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            margin=dict(t=50, l=50, r=50, b=50)
        )
        
        return fig

    return app
