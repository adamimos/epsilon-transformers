import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNWrapper(nn.Module):
    def __init__(self, rnn, output_layer):
        super().__init__()
        self.rnn = rnn
        self.output_layer = output_layer
        self.vocab_size = output_layer.out_features  # Store vocab size for one-hot encoding
    
    def forward(self, x):
        # Convert input tokens to one-hot vectors
        # x shape: (batch_size, seq_length)
        # one_hot shape: (batch_size, seq_length, vocab_size)
        one_hot = F.one_hot(x.to(torch.int64), num_classes=self.vocab_size).float()
        
        output, hidden = self.rnn(one_hot)
        return self.output_layer(output)

    def forward_with_all_states(self, x):
        """Forward pass that returns states at all layers and timepoints"""
        batch_size, seq_len = x.shape
        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        
        # Input processing
        one_hot = F.one_hot(x.to(torch.int64), num_classes=self.vocab_size).float()
        
        # Storage for all states
        all_layer_states = torch.zeros(num_layers, batch_size, seq_len, hidden_size, device=x.device)
        
        # Initialize hidden state
        if isinstance(self.rnn, nn.LSTM):
            h = torch.zeros(num_layers, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_layers, batch_size, hidden_size, device=x.device)
            hidden = (h, c)
        else:  # GRU or RNN
            hidden = torch.zeros(num_layers, batch_size, hidden_size, device=x.device)
        
        # Process each timestep
        for t in range(seq_len):
            # Get single timestep input
            timestep_input = one_hot[:, t:t+1, :]
            
            # Run RNN for single timestep
            output, new_hidden = self.rnn(timestep_input, hidden)
            
            # Store hidden states
            if isinstance(self.rnn, nn.LSTM):
                # For LSTM, store h states
                all_layer_states[:, :, t, :] = new_hidden[0]
            else:
                # For GRU/RNN, store hidden states
                all_layer_states[:, :, t, :] = new_hidden
            
            # Update hidden for next timestep
            hidden = new_hidden
        
        # Get final output sequence
        output, _ = self.rnn(one_hot, None)
        logits = self.output_layer(output)
        
        return logits, {
            'layer_states': all_layer_states,  # [num_layers, batch, seq, hidden]
            'final_output': output,  # [batch, seq, hidden]
            'logits': logits  # [batch, seq, vocab]
        }
    

def create_RNN(config, vocab_size, device):
    rnn_type = config['model_config'].get('rnn_type', 'LSTM').upper()
    rnn_class = getattr(nn, rnn_type)
    
    rnn = rnn_class(
        input_size=vocab_size,
        hidden_size=config['model_config']['hidden_size'],
        num_layers=config['model_config']['num_layers'],
        dropout=config['model_config'].get('dropout', 0.0),
        batch_first=True,
        bidirectional=config['model_config'].get('bidirectional', False)
    ).to(device)
    
    output_size = config['model_config']['hidden_size'] * 2 if config['model_config'].get('bidirectional', False) else config['model_config']['hidden_size']
    output_layer = nn.Linear(output_size, vocab_size).to(device)
    
    return RNNWrapper(rnn, output_layer)