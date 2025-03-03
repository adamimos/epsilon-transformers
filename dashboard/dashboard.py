"""
Simplified Activation Analysis Dashboard

Focuses on key visualizations for activation analysis:
1. Normalized Distance vs. Checkpoint (all layers)
2. Loss vs. Normalized Distance
"""
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import ActivationAnalysisLoader
import numpy as np

# Set page config
st.set_page_config(
    page_title="Activation Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize data loader with session state
if 'loader' not in st.session_state:
    st.session_state.loader = None

# Function to initialize the loader
def initialize_loader():
    """Initialize the data loader."""
    try:
        loader = ActivationAnalysisLoader()
        st.session_state.loader = loader
        return loader
    except Exception as e:
        st.error(f"Error initializing data loader: {str(e)}")
        return None

# Title
st.title("Activation Analysis Dashboard")

# Initialize loader if not already done
if st.session_state.loader is None:
    initialize_loader()

# Get the loader from session state
loader = st.session_state.loader

# Selection panel
st.sidebar.header("Data Selection")

# Only proceed if loader is initialized
if loader is not None:
    # Sweep selection
    available_sweeps = loader.sweeps
    if not available_sweeps:
        st.warning("No sweeps found. Please check that your analysis directory is correct.")
    else:
        selected_sweep = st.sidebar.selectbox(
            "Select Sweep ID",
            available_sweeps,
            index=0 if available_sweeps else None,
        )
        
        # Only continue if a sweep is selected
        if selected_sweep:
            runs = loader.get_runs_in_sweep(selected_sweep)
            
            if not runs:
                st.warning(f"No runs found in sweep {selected_sweep}.")
            else:
                # Run selection
                selected_run = st.sidebar.selectbox(
                    "Select Run",
                    runs,
                )
                
                # Display selected run info
                if selected_run:
                    st.sidebar.markdown(f"**Selected Run:** {selected_run}")
                    
                    # Parse run components for display
                    components = loader.parse_run_id_components(selected_run)
                    st.sidebar.markdown("#### Run Details")
                    for key, value in components.items():
                        if key != 'run_id':
                            st.sidebar.markdown(f"**{key}:** {value}")
                    
                    # Visualization options
                    st.sidebar.markdown("#### Visualization Options")
                    use_normalized = st.sidebar.checkbox(
                        "Use Normalized Values", 
                        value=True,
                        help="Normalize norm_dist by random baseline means"
                    )
                    
                    # Load data
                    with st.spinner("Loading and processing data..."):
                        try:
                            # Load processed regression results
                            regression_df = loader.load_processed_regression_results(selected_sweep, selected_run)
                            
                            # Check if normalized data is available
                            has_normalized = 'norm_dist_normalized' in regression_df.columns
                            if use_normalized and not has_normalized:
                                st.warning("Normalized data not available. Using raw norm_dist values.")
                                use_normalized = False
                            
                            # Determine which norm_dist column to use
                            norm_dist_col = 'norm_dist_normalized' if use_normalized and has_normalized else 'norm_dist'
                            
                            # Try to load loss data (if available)
                            try:
                                loss_df = loader.load_loss_data(selected_sweep, selected_run)
                                has_loss = True
                            except FileNotFoundError:
                                has_loss = False
                            
                            # Main content area
                            
                            # Filter to select target - moved to top level
                            targets = sorted(regression_df['target'].unique())
                            selected_target = st.selectbox(
                                "Select Target",
                                options=targets,
                            )
                            
                            # Filter data for the selected target
                            target_data = regression_df[regression_df['target'] == selected_target]
                            
                            # Create a mapping of layer indices to names and ensure proper sorting
                            layer_mapping = target_data[['layer_idx', 'layer_name']].drop_duplicates().set_index('layer_idx')
                            layer_mapping = layer_mapping.sort_index()
                            
                            # Create a custom layer ordering function
                            def get_layer_order(layer_name, layer_idx):
                                """
                                Custom ordering function that places input layer first,
                                then sorts remaining layers by index.
                                """
                                # Check if this is an input layer (using string match)
                                is_input = any(keyword in layer_name.lower() for keyword in 
                                              ['input', 'embed', 'hook_resid_pre'])
                                
                                # Return a tuple for sorting - first element determines primary sort order
                                # -1 for input layer (to place it first), layer_idx for others
                                return (-1 if is_input else layer_idx, layer_idx)
                            
                            # Normalized Distance by Layer plot
                            st.header(f"{'Normalized ' if use_normalized and has_normalized else ''}Distance vs. Checkpoint by Layer")
                            
                            # Group by layer and checkpoint
                            pivot_df = target_data.pivot_table(
                                values=norm_dist_col,
                                index='checkpoint',
                                columns=['layer_idx', 'layer_name'],
                                aggfunc='mean'
                            )
                            
                            # Create a list of column tuples (layer_idx, layer_name) for custom sorting
                            layer_columns = list(pivot_df.columns)
                            
                            # Sort the columns using our custom ordering function
                            sorted_columns = sorted(layer_columns, 
                                                  key=lambda x: get_layer_order(x[1], x[0]))
                            
                            # Reindex the pivot table with the sorted columns
                            pivot_df = pivot_df[sorted_columns]
                            
                            # Create nice display names for the layers
                            display_names = []
                            for idx, name in sorted_columns:
                                # Check if this is an input layer for special formatting
                                if any(keyword in name.lower() for keyword in 
                                      ['input', 'embed', 'hook_resid_pre']):
                                    display_names.append(f"Input: {name} (Layer {idx})")
                                else:
                                    display_names.append(f"{name} (Layer {idx})")
                            
                            # Reset the column multi-index with our display names
                            pivot_df.columns = display_names
                            
                            # Create a custom continuous color scale based on the sorted order
                            # Input layer gets a distinct color, then continuous gradient for others
                            color_scale = px.colors.sequential.Viridis
                            
                            # Create multi-line plot with continuous coloring
                            fig1 = px.line(
                                pivot_df,
                                markers=True,
                                title=f"{'Normalized ' if use_normalized and has_normalized else ''}Distance vs. Checkpoint Across Layers ({selected_target})",
                                labels={
                                    'value': 'Normalized Distance' if use_normalized and has_normalized else 'Normalized Distance', 
                                    'index': 'Checkpoint', 
                                    'variable': 'Layer'
                                },
                                color_discrete_sequence=color_scale,
                            )
                            
                            # Update the legend title
                            fig1.update_layout(legend_title_text="Layer")
                            
                            # Add a reference line at y=1.0 for normalized values
                            if use_normalized and has_normalized:
                                fig1.add_hline(y=1.0, line_dash="dash", line_color="red", 
                                               annotation_text="Random Baseline", 
                                               annotation_position="top right")
                                
                            st.plotly_chart(fig1, use_container_width=True)

                            # Loss vs. norm_dist plot (if loss data is available)
                            if has_loss:
                                st.header("Loss vs. Normalized Distance")
                                
                                # Filter out non-numeric checkpoints like "RANDOM_0"
                                numeric_data = target_data[target_data['checkpoint'].astype(str).str.isdigit()]
                                
                                if not numeric_data.empty and 'num_tokens_seen' in loss_df.columns:
                                    # Convert checkpoint to int
                                    numeric_data['checkpoint'] = numeric_data['checkpoint'].astype(int)
                                    
                                    # Get unique checkpoint values and tokens_seen values
                                    checkpoint_values = sorted(numeric_data['checkpoint'].unique())
                                    tokens_seen_values = sorted(loss_df['num_tokens_seen'].unique())
                                    
                                    # Identify mapping between checkpoints and tokens_seen
                                    if len(checkpoint_values) == len(tokens_seen_values):
                                        # Direct 1:1 mapping
                                        checkpoint_to_tokens = dict(zip(checkpoint_values, tokens_seen_values))
                                        numeric_data['tokens_seen'] = numeric_data['checkpoint'].map(checkpoint_to_tokens)
                                    else:
                                        # Try to find a scaling factor
                                        best_factor = None
                                        possible_factors = []
                                        
                                        for c in checkpoint_values:
                                            for t in tokens_seen_values:
                                                if c > 0:  # Avoid division by zero
                                                    factor = t / c
                                                    if abs(factor - round(factor)) < 0.01:
                                                        possible_factors.append((round(factor), c, t))
                                        
                                        if possible_factors:
                                            # Count occurrences of each factor
                                            factor_counts = {}
                                            for f, _, _ in possible_factors:
                                                if f not in factor_counts:
                                                    factor_counts[f] = 0
                                                factor_counts[f] += 1
                                            
                                            # Find the most common factor
                                            best_factor = max(factor_counts.items(), key=lambda x: x[1])[0]
                                            numeric_data['tokens_seen'] = numeric_data['checkpoint'] * best_factor
                                    
                                    # If we successfully mapped checkpoints to tokens
                                    if 'tokens_seen' in numeric_data.columns:
                                        # Merge with loss data for each layer
                                        merged_data = pd.merge(
                                            numeric_data,
                                            loss_df[['num_tokens_seen', 'val_loss_mean']],
                                            left_on='tokens_seen',
                                            right_on='num_tokens_seen',
                                            how='inner'
                                        )
                                        
                                        if not merged_data.empty and len(merged_data) > 1:
                                            # Use the same custom layer ordering as the previous plot
                                            merged_data['sort_key'] = merged_data.apply(
                                                lambda row: get_layer_order(row['layer_name'], row['layer_idx']), 
                                                axis=1
                                            )
                                            
                                            # Add a column for nice display names
                                            merged_data['display_name'] = merged_data.apply(
                                                lambda row: f"Input: {row['layer_name']} (Layer {row['layer_idx']})" 
                                                        if any(keyword in row['layer_name'].lower() for keyword in ['input', 'embed', 'hook_resid_pre'])
                                                        else f"{row['layer_name']} (Layer {row['layer_idx']})",
                                                axis=1
                                            )
                                            
                                            # Sort by our custom order
                                            merged_data = merged_data.sort_values('sort_key')
                                            
                                            # Create the scatter plot with all layers and coloring
                                            fig2 = px.scatter(
                                                merged_data,
                                                x=norm_dist_col,
                                                y='val_loss_mean',
                                                color='display_name',
                                                hover_data=['checkpoint', 'num_tokens_seen', 'layer_name'],
                                                title=f"Validation Loss vs. {'Normalized ' if use_normalized and has_normalized else ''}Distance ({selected_target})",
                                                labels={
                                                    norm_dist_col: 'Normalized Distance' if use_normalized and has_normalized else 'Normalized Distance', 
                                                    'val_loss_mean': 'Validation Loss',
                                                    'display_name': 'Layer'
                                                },
                                                color_discrete_sequence=px.colors.sequential.Viridis,
                                            )
                                            
                                            # Update the legend title
                                            fig2.update_layout(legend_title_text="Layer")
                                            
                                            st.plotly_chart(fig2, use_container_width=True)
                                        else:
                                            st.info("Not enough data points to create the Loss vs. Distance plot.")
                                    else:
                                        st.info("Could not establish a mapping between checkpoints and tokens seen.")
                                else:
                                    st.info("Loss data or numeric checkpoint data not available for this visualization.")
                            
                            # Add Loss vs. Checkpoint plot
                            if has_loss:
                                st.header("Loss vs. Checkpoint")
                                
                                if 'num_tokens_seen' in loss_df.columns:
                                    # Sort loss data by tokens seen for proper plotting
                                    sorted_loss = loss_df.sort_values('num_tokens_seen')
                                    
                                    # Create the loss vs checkpoint plot
                                    fig3 = px.line(
                                        sorted_loss,
                                        x='num_tokens_seen',
                                        y=['train_loss_mean', 'val_loss_mean'],
                                        markers=True,
                                        title=f"Training and Validation Loss Over Time",
                                        labels={
                                            'num_tokens_seen': 'Tokens Seen (Checkpoint)',
                                            'value': 'Loss',
                                            'variable': 'Loss Type'
                                        },
                                    )
                                    
                                    # Update the legend to have more readable names
                                    fig3.for_each_trace(lambda t: t.update(
                                        name={'train_loss_mean': 'Training Loss', 
                                              'val_loss_mean': 'Validation Loss'}[t.name]
                                    ))
                                    
                                    # Add vertical lines at checkpoint positions if we have the mapping
                                    if 'tokens_seen' in locals() and 'merged_data' in locals() and not merged_data.empty:
                                        checkpoints = merged_data['tokens_seen'].unique()
                                        for checkpoint in checkpoints:
                                            fig3.add_vline(
                                                x=checkpoint, 
                                                line_dash="dash", 
                                                line_color="rgba(0,0,0,0.3)"
                                            )
                                    
                                    # Update the layout for better visibility
                                    fig3.update_layout(
                                        legend_title_text="",
                                        xaxis_title="Tokens Seen (Checkpoint positions shown as dashed lines)"
                                    )
                                    
                                    st.plotly_chart(fig3, use_container_width=True)
                                    
                                    # Add a small table showing the checkpoint to tokens mapping if available
                                    if 'tokens_seen' in locals() and 'merged_data' in locals() and not merged_data.empty:
                                        with st.expander("Checkpoint to Tokens Mapping"):
                                            mapping_df = merged_data[['checkpoint', 'tokens_seen']].drop_duplicates().sort_values('checkpoint')
                                            mapping_df.columns = ['Checkpoint', 'Tokens Seen']
                                            st.dataframe(mapping_df)
                                else:
                                    st.info("Loss data does not contain token information needed for this plot.")
                            
                            # Display data tables in expandable sections
                            with st.expander("View Processed Regression Data"):
                                st.dataframe(regression_df.head(20))
                            
                            if has_loss:
                                with st.expander("View Loss Data"):
                                    st.dataframe(loss_df.head(20))
                            
                            # Random baseline mean values table - moved to the bottom
                            if has_normalized:
                                st.subheader("Random Baseline Mean Values")
                                random_means = target_data.groupby(['layer_idx', 'layer_name'])['random_mean_norm_dist'].mean().reset_index()
                                
                                # Add a column for custom sorting
                                random_means['sort_key'] = random_means.apply(
                                    lambda row: get_layer_order(row['layer_name'], row['layer_idx']), axis=1
                                )
                                
                                # Sort by our custom order
                                random_means = random_means.sort_values('sort_key')
                                
                                # Add a type column for clarity
                                random_means['layer_type'] = random_means['layer_name'].apply(
                                    lambda name: "Input Layer" if any(keyword in name.lower() for keyword in 
                                                                  ['input', 'embed', 'hook_resid_pre']) 
                                              else "Hidden Layer"
                                )
                                
                                # Rename columns for better display and drop sort key
                                display_df = random_means[['layer_idx', 'layer_name', 'layer_type', 'random_mean_norm_dist']]
                                display_df.columns = ['Layer Index', 'Layer Name', 'Layer Type', 'Random Mean']
                                
                                st.dataframe(display_df, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error loading data: {str(e)}")
                            st.exception(e)
else:
    st.warning("Data loader not initialized. Please check the console for errors.")

# Footer
st.divider()
st.caption("Activation Analysis Dashboard") 