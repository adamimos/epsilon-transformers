import argparse
import yaml
import os
import json
import numpy as np
from epsilon_transformers.training.dataloader import get_dataloader_and_loss_lower_bound_from_process
import torch

def load_process_data(config, process_dir):
    process_config = config['process_config']
    n_ctx = config['model_config']['n_ctx']
    bos = config['train_config']['bos']
    
    process_string = get_process_string(process_config, n_ctx, bos)
    data_dir = os.path.join(process_dir, process_string)
    
    if not os.path.exists(data_dir):
        return None
    
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    loss_lower_bound = np.load(os.path.join(data_dir, 'loss_lower_bound.npy'))
    transformer_inputs = np.load(os.path.join(data_dir, 'transformer_inputs.npy'))
    probs = np.load(os.path.join(data_dir, 'probs.npy'))
    
    return {
        'metadata': metadata,
        'loss_lower_bound': loss_lower_bound,
        'transformer_inputs': transformer_inputs,
        'probs': probs
    }

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def get_process_string(process_config, n_ctx, bos):
    process_string = "_".join(f"{key}_{value}" for key, value in process_config.items() if key != 'name')
    return f"{process_config['name']}_ctx{n_ctx}_bos{bos}_{process_string}"

def compare_metadata(metadata1, metadata2):
    return (metadata1['process_config'] == metadata2['process_config'] and
            metadata1['n_ctx'] == metadata2['n_ctx'] and
            metadata1['d_vocab'] == metadata2['d_vocab'] and
            metadata1['bos'] == metadata2['bos'])

def generate_and_save_data(config, output_dir):
    for process_config in config['sweep_config']['process_config']:
        process_name = process_config['name']
        
        for n_ctx in config['sweep_config']['model_config'].get('n_ctx', [config['model_config']['n_ctx']]):
            dataloader, loss_lower_bound, d_vocab = get_dataloader_and_loss_lower_bound(
                process_params=process_config,
                n_ctx=n_ctx,
                bos=config['train_config']['bos'],
                batches_per_epoch=1,
                batch_size=1,
                device='cpu'  # Generate on CPU for consistency
            )

            # Save the data
            process_string = get_process_string(process_config, n_ctx, config['train_config']['bos'])
            data_dir = os.path.join(output_dir, process_string)
            os.makedirs(data_dir, exist_ok=True)

            np.save(os.path.join(data_dir, 'loss_lower_bound.npy'), loss_lower_bound.numpy())
            
            # Save dataloader data
            transformer_inputs = dataloader.transformer_inputs.numpy() if torch.is_tensor(dataloader.transformer_inputs) else dataloader.transformer_inputs
            probs = dataloader.probs.numpy() if torch.is_tensor(dataloader.probs) else dataloader.probs
            np.save(os.path.join(data_dir, 'transformer_inputs.npy'), transformer_inputs)
            np.save(os.path.join(data_dir, 'probs.npy'), probs)
            # Save metadata
            metadata = {
                'process_config': process_config,
                'n_ctx': n_ctx,
                'd_vocab': d_vocab,
                'bos': config['train_config']['bos']
            }
            with open(os.path.join(data_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)

            print(f"Data generated and saved for {process_name}, n_ctx={n_ctx}")

def main():
    parser = argparse.ArgumentParser(description='Generate data based on experiment configuration.')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config['global_config']['process_dir']

    generate_and_save_data(config, output_dir)

if __name__ == "__main__":
    main()