import plotly.graph_objects as go
import webcolors
from typing import List, Dict
from dataclasses import dataclass
import numpy as np

from epsilon_transformers.analysis.training_dynamics.weight_metrics import AttnCircuit, AttnCircuitsMeasurements, Metric
from epsilon_transformers.analysis.training_dynamics.activation_metrics import Extracted, activation_effective_dimension

# TODO: Test plot_attn_circuit_measurements()
# TODO: Rename Line to something sensible
# TODO: Implement more sensible error bounds on from_extracted_activations()

@dataclass
class Line:
    name: str
    color: str
    x: List[float]
    y: List[float]
    upper_bound: List[float]
    lower_bound: List[float]

    def __post_init__(self):
        # Check that the color is valid and re-init it
        self.color = webcolors.name_to_rgb(self.color)
        
        # Check that all lenghts are the same
        lengths = [len(self.x), len(self.y), len(self.upper_bound), len(self.lower_bound)]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All lists in Line must have the same length")
    
    @classmethod
    def from_attn_circuit_measurement(cls, circuits: List[AttnCircuitsMeasurements], color: str, circuit_type: AttnCircuit, metric: Metric) -> "Line":
        assert all([circuit.model_name == circuits[0].model_name for circuit in circuits]), "Circuit measurements must come from the same model"
        assert all([getattr(circuit, circuit_type) is not None for circuit in circuits]), "circuit_type must be present in all AttnCircuitMeasurements"

        index = f'{circuit_type}.{metric}'
        return Line(
            name=circuits[0].model_name,
            color=color,
            x=[circuit.num_tokens_seen for circuit in circuits],
            y=[circuit.mean_reducer(index) for circuit in circuits],
            lower_bound=[circuit.tensor_min(index) for circuit in circuits],
            upper_bound=[circuit.tensor_max(index) for circuit in circuits]
        )
    
    @classmethod
    def from_extracted_activations(cls, extractions: List[Extracted], module_name: str, color: str, metric: Metric) -> 'Line':
        if metric == 'effective_rank':
            raise NotImplementedError("Effective Rank is not yet implemented")
        out = [activation_effective_dimension(x.activations)[module_name] for x in extractions]
        
        return Line(
            name=extractions[0].model_name,
            color=color,
            x=[extraction.num_tokens_seen for extraction in extractions],
            y=out,
            lower_bound=out,
            upper_bound=out,
        )
    
def continuous_error_boundary_plot(
    lines: List[Line], 
    title: str,
    x_axis_title: str,
    y_axis_title: str,
) -> go.Figure:
    fig = go.Figure()
    for line in lines:
        # Add the main line
        fig.add_trace(go.Scatter(x=line.x, y=line.y, mode='lines', line=dict(color=f'rgb({line.color[0]}, {line.color[1]}, {line.color[2]})'), name=line.name))

        # Add the upper boundary
        fig.add_trace(go.Scatter(x=line.x, y=line.upper_bound, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))

        # Add the lower boundary
        fig.add_trace(go.Scatter(x=line.x, y=line.lower_bound, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))

        # Add the shaded region between upper and lower boundaries
        fig.add_trace(go.Scatter(x=np.concatenate([line.x, line.x[::-1]]),
                                 y=np.concatenate([line.upper_bound, line.lower_bound[::-1]]),
                                 fill='toself', fillcolor=f'rgba({line.color[0]}, {line.color[1]}, {line.color[2]}, 0.2)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 showlegend=False))
    # Update layout
    fig.update_layout(title=title, xaxis_title=x_axis_title, yaxis_title=y_axis_title)
    
    return fig

def plot_attn_circuit_measurements(
    measurements_color_dict: Dict[str, List[AttnCircuitsMeasurements]], 
    circuit_type: AttnCircuit, 
    metric: Metric,
    title: str,
    x_axis_title: str,
    y_axis_title: str
) -> go.Figure:
  lines = [Line.from_attn_circuit_measurement(measurements, color, circuit_type, metric) for color, measurements in measurements_color_dict.items()]
  return continuous_error_boundary_plot(lines=lines, title=title, x_axis_title=x_axis_title, y_axis_title=y_axis_title)
        
if __name__ == "__main__":
  import numpy as np
  lines = [
      Line(name='Sin Line', color='red', x=np.linspace(0, 10, 100), y=np.sin(np.linspace(0, 10, 100)),
          upper_bound=np.sin(np.linspace(0, 10, 100)) + 0.2, lower_bound=np.sin(np.linspace(0, 10, 100)) - 0.2),
      Line(name='Cos Line', color='blue', x=np.linspace(0, 10, 100), y=np.cos(np.linspace(0, 10, 100)),
          upper_bound=np.cos(np.linspace(0, 10, 100)) + 0.1, lower_bound=np.cos(np.linspace(0, 10, 100)) - 0.1),
  ]

  fig = continuous_error_boundary_plot(lines, 'the graphy graph', 'foo', 'bar')
  fig.show()