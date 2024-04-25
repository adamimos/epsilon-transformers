from typing import Tuple
import torch
from jaxtyping import Float

class RNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.input_to_hidden = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.input_to_output = torch.nn.Linear(input_size + hidden_size, output_size)
        self.relu = torch.nn.ReLU()
    
    def forward(self, input_tensor: Float[torch.Tensor, "input_size vocab_size"], hidden_tensor: Float[torch.Tensor, "hidden_size"]) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((input_tensor, hidden_tensor), dim=1)
        hidden = self.input_to_hidden(combined)
        output = self.input_to_output(combined)
        output = self.relu(output)
        
        return hidden, output
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)