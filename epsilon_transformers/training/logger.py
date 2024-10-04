import csv
import json
import os
from typing import Dict, Any, List

import torch
import torch.nn as nn
import numpy as np
import wandb

class StructuredLogger:
    def __init__(self, save_dir: str):
        self.base_dir = save_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.csv_file = open(os.path.join(self.base_dir, 'log.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.json_file = open(os.path.join(self.base_dir, 'log.json'), 'w')

    def log(self, data: Dict[str, Any]):
        # Flatten the dictionary for CSV
        flat_data = self._flatten_dict(data)
        if not hasattr(self, 'csv_headers'):
            self.csv_headers = list(flat_data.keys())
            self.csv_writer.writerow(self.csv_headers)
        self.csv_writer.writerow([flat_data.get(h, '') for h in self.csv_headers])

        # Write to JSON
        json.dump(data, self.json_file)
        self.json_file.write('\n')
        self.json_file.flush()

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def log_epoch(self, epoch: int, num_tokens_seen: int, train_losses: List[float], val_losses: List[float], learning_rate: float):
        loss_file_path = os.path.join(self.base_dir, 'loss.csv')

        # if train_losses is None set it to inf, same for val_losses
        train_losses = [np.inf for _ in val_losses] if train_losses is None else train_losses
        val_losses = [np.inf for _ in val_losses] if val_losses is None else val_losses
        
        # Check if the file exists, if not, create it and write the header
        if not os.path.exists(loss_file_path):
            with open(loss_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                header = ['epoch', 'num_tokens_seen']
                for i in range(len(train_losses)):
                    header.extend([f'train_loss_ctx{i}', f'val_loss_ctx{i}'])
                header.extend(['train_loss_mean', 'val_loss_mean', 'learning_rate'])
                writer.writerow(header)
        
        # Log to wandb
        wandb_log = {
            'epoch': epoch,
            'num_tokens_seen': num_tokens_seen,
            'learning_rate': learning_rate
        }
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            wandb_log[f'train_loss_ctx{i}'] = train_loss
            wandb_log[f'val_loss_ctx{i}'] = val_loss
        wandb_log['train_loss_mean'] = np.mean(train_losses)
        wandb_log['val_loss_mean'] = np.mean(val_losses)
        
        wandb.log(wandb_log)
        # Append the new data
        with open(loss_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [epoch, num_tokens_seen]
            for train_loss, val_loss in zip(train_losses, val_losses):
                row.extend([train_loss, val_loss])
            row.extend([np.mean(train_losses), np.mean(val_losses), learning_rate])
            writer.writerow(row)
    
    def save_model_checkpoint(self, model: nn.Module, name: str):
        torch.save(model.state_dict(), os.path.join(self.base_dir, f'{name}.pt'))

    def close(self):
        self.csv_file.close()
        self.json_file.close()