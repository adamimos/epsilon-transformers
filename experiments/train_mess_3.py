from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.training.configs.training_configs import LoggingConfig, OptimizerConfig, PersistanceConfig, ProcessDatasetConfig, TrainConfig
from epsilon_transformers.training.train import train_model

# TODO: Assert model_config.d_vocab == process.d_vocab
# TODO: Assert that process_params and process type are set correctly
# TODO: Change test_split into num_of_eval_tokens and total_num_of_train_tokens (or better yet, eval on the set of all possible continuations)
# TODO: Pretty print 'This is the log'

model_config = RawModelConfig(
    d_vocab=3,
    d_model=64,
    n_ctx=10,
    d_head=8,
    n_head=1,
    d_mlp=12,
    n_layers=4,
)

optimizer_config = OptimizerConfig(
    optimizer_type='sgd',
    learning_rate=1e-2,
    weight_decay=0
)

dataset_config = ProcessDatasetConfig(
    process='mess3',
    process_params={'x': 0.5, 'a': 0.85},
    batch_size=64,
    num_tokens=500000000000000,
    test_split=0.00000000015
)

persistance_config = PersistanceConfig(
    location='s3',
    collection_location='lucas-mess3-test',
    checkpoint_every_n_tokens=100000
)

mock_config = TrainConfig(
    model=model_config,
    optimizer=optimizer_config,
    dataset=dataset_config,
    persistance=persistance_config,
    logging=LoggingConfig(project_name="lucas-mess3-test", wandb=True),
    verbose=True,
    seed=42
)

if __name__ == "__main__":
    train_model(mock_config)
