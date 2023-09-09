

cartpole_hyperparameters = {
    "render_mode": "rgb_array",
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": "CartPole-v1",
}

pixelcopter_hyperparameters = {
    "render_mode": "rgb_array",
    "h_size": 64,
    "n_training_episodes": 50000,
    "n_evaluation_episodes": 10,
    "max_t": 10000,
    "gamma": 0.99,
    "lr": 1e-4,
    "env_id": "Pixelcopter-PLE-v0",
}

train_config = cartpole_hyperparameters
