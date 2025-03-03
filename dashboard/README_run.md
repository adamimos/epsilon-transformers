# Running the Activation Analysis Dashboard

This document provides instructions for running the activation analysis dashboard.

## Prerequisites

Before running the dashboard, ensure you have the following:

1. Python 3.8 or higher installed
2. The analysis data directory with sweep results
3. Required Python packages (see below)

## Installation

1. Install the required packages:

```bash
cd dashboard
pip install -r requirements.txt
```

## Running the Dashboard

From the project root directory, run:

```bash
cd dashboard
streamlit run dashboard.py
```

This will start the Streamlit server and open the dashboard in your default web browser. If it doesn't open automatically, the terminal will display a URL (typically http://localhost:8501) that you can open in your browser.

## Using the Dashboard

1. **Configure Data Source**:
   - If the dashboard doesn't automatically find your analysis directory, you can specify it manually in the "Analysis Directory Path" field
   - Click "Initialize/Refresh Data Loader" to reload with the specified path
   - The dashboard will display the number of sweeps found in the specified directory

2. **Select Data**:
   - Use the sidebar to select a sweep ID
   - Then select a specific run to analyze

3. **Explore Visualizations**:
   - **Normalized Distance vs. Checkpoint**: Shows how the normalized distance metric changes across checkpoints for a selected layer
   - **Loss vs. Normalized Distance**: Visualizes the relationship between validation loss and normalized distance
   - **Normalized Distance by Layer**: Compares normalized distance across all layers

4. **View Data**:
   - Expandable sections at the bottom provide access to the raw data tables

## Troubleshooting

- **Missing Data**: If loss data is unavailable, some visualizations may show informational messages
- **Data Loading Errors**: Check that the analysis directory structure matches the expected format
- **Performance Issues**: For large datasets, the initial loading may take some time

### Debug Script

If you're having trouble with the dashboard finding your data, you can run the included debug script:

```bash
cd dashboard
python debug_data_loader.py --analysis-dir /path/to/your/analysis/directory
```

This script will:
1. Test the data loader with the specified path
2. Print information about what it finds
3. Try alternative paths to see if your data is in a different location

The output from this script can help identify where your analysis data is located and if there are any issues accessing it.

## Extending the Dashboard

To add new visualizations or features:

1. Edit `dashboard.py` to add new components
2. Update the data loader in `data_loader.py` if needed
3. Restart the Streamlit server to see your changes 