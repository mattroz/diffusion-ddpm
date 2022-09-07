from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size = 256
    image_channels = 3
    train_batch_size = 6
    eval_batch_size = 6
    num_epochs = 150
    start_epoch = 0
    learning_rate = 2e-5
    diffusion_timesteps = 1000
    save_image_epochs = 5
    save_model_epochs = 5
    dataset = 'alkzar90/croupier-mtg-dataset'
    output_dir = f'models/{dataset.split("/")[-1]}'
    device = "cuda"
    seed = 0
    resume = None


training_config = TrainingConfig()