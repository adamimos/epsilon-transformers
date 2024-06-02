from dataclasses import dataclass
import pathlib
import torch
from typing import Iterable, List
from transformer_lens import HookedTransformer
import plotly.graph_objects as go


from epsilon_transformers.persistence import state_dict_to_model_config
from epsilon_transformers.analysis.training_dynamics.weight_metrics import AttnCircuit, AttnCircuitsMeasurements, Metric
from epsilon_transformers.visualization.training_dynamics import Line, continuous_error_boundary_plot

def _model_iterator_factory(dir: pathlib.Path, device: torch.device = torch.device('cpu')) -> Iterable[HookedTransformer]:
    assert dir.is_dir()
    assert dir.exists()

    files = [file for file in dir.glob('*.pt') if file.stem.isnumeric()]
    tokens_trained = sorted([int(str(x.stem)) for x in files if str(x)[-3:] == '.pt'])
    files_sorted = [f"{tokens}.pt" for tokens in tokens_trained]

    def _iterator(files: List[str]):
        for file in files:
            state_dict = torch.load(dir / file)
            config = state_dict_to_model_config(state_dict=state_dict)
            model = config.to_hooked_transformer(device=device)
            model.load_state_dict(state_dict=state_dict)
            yield model

    return _iterator(files=files_sorted)

@dataclass
class PlotConfig:
    colors: List[str]
    circuits: List[AttnCircuit]
    metric: Metric
    title: str
    x_axis_title: str
    y_axis_title: str

def _plots_from_models(model_iterator: Iterable[HookedTransformer], plot_config: PlotConfig) -> go.Figure:
    measurement_list = [AttnCircuitsMeasurements.from_model(model=model, circuits=plot_config.circuits) for model in model_iterator]
    lines = [Line.from_attn_circuit_measurement(circuits=measurement_list, color=color, circuit_type=circuit, metric=plot_config.metric) for circuit, color in zip(plot_config.circuits, plot_config.colors)]
    return continuous_error_boundary_plot(lines=lines, title=plot_config.title, x_axis_title=plot_config.x_axis_title, y_axis_title=plot_config.y_axis_title)

if __name__ == "__main__":
    model_iterator = _model_iterator_factory(dir=pathlib.Path('./experiments/models/rrxor'))
    config = PlotConfig(colors=['red', 'blue'], circuits=['qk', 'ov'], metric='effective_dimension', title='RRXOR Effective Dimensionality (mean line, min-max error bars)', x_axis_title='Num Tokens', y_axis_title='Effective Dimensionality')
    fig = _plots_from_models(model_iterator=model_iterator, plot_config=config)
    fig.show()