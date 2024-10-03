# launcher.py

import argparse
import itertools
import subprocess
import yaml
import os
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

def run_experiment(run_config):
    cmd = [
        'python', 'scripts/train.py',
        '--config', run_config['config_path'],
    ]
    subprocess.run(cmd)


def create_config_sweep(config):
    global_config = config.get('global_config', {})

    model_config = config.get('model_config', {})
    train_config = config.get('train_config', {})

    sweep_config = config.get('sweep_config', {})
    sweep_train_config = sweep_config.get('train_config', {})
    sweep_model_config = sweep_config.get('model_config', {})
    sweep_process_config = sweep_config.get('process_config', {})
    # use itertools.product to create the combinations
    model_config_combinations = [dict(zip(sweep_model_config.keys(), combination)) for combination in itertools.product(*sweep_model_config.values())]
    train_config_combinations = [dict(zip(sweep_train_config.keys(), combination)) for combination in itertools.product(*sweep_train_config.values())]
    # now append the constant values from the model_config and train_config
    for cfg in model_config_combinations:
        cfg.update(model_config)
        if 'd_model' not in cfg:
            cfg['d_model'] = cfg['d_head'] * cfg['n_heads']
        if 'd_mlp' not in cfg:
            cfg['d_mlp'] = 4 * cfg['d_model']
    for cfg in train_config_combinations:
        cfg.update(train_config)

    # Create a combined iterator
    combined_config_iter = itertools.product(model_config_combinations, train_config_combinations, sweep_process_config)

    # Create the final iterator of dict of dicts
    config_sweep_iter = (
        {
            'global_config': global_config,
            'model_config': model_cfg,
            'train_config': train_cfg,
            'process_config': process_cfg
        }
        for model_cfg, train_cfg, process_cfg in combined_config_iter
    )
    return config_sweep_iter

def main():
    parser = argparse.ArgumentParser(description='Launch multiple training jobs.')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment configuration file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Generate all combinations for the sweep
    sweep_params = create_config_sweep(config)

    sweep_id = config.get('sweep_id', datetime.now().strftime("%Y%m%d%H%M%S"))

    os.makedirs(f"{config['global_config']['output_dir']}/{sweep_id}", exist_ok=True)

    for i, cfg in enumerate(sweep_params):
        experiment_name = f"run_{i}_L{cfg['model_config']['n_layers']}_H{cfg['model_config']['n_heads']}_DH{cfg['model_config']['d_head']}_DM{cfg['model_config']['d_model']}_{cfg['process_config']['name']}"
        config_path = f"{config['global_config']['output_dir']}/{sweep_id}/config_{experiment_name}.yaml"
        cfg['run_id'] = experiment_name
        cfg['global_config']['sweep_id'] = sweep_id
        cfg['config_path'] = config_path
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f)

        run_experiment(cfg)

if __name__ == "__main__":
    main()
