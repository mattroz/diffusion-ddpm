import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PIL import Image
from pathlib import Path

def broadcast(values, broadcast_to):
    values = values.flatten()

    while len(values.shape) < len(broadcast_to.shape):
        values = values.unsqueeze(-1)

    return values


def postprocess(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return images


def create_images_grid(images, rows, cols):
    images = [Image.fromarray(image) for image in images]
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def create_sampling_animation(model, pipeline, config, interval=5):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    noisy_sample = torch.randn(
        config.eval_batch_size,
        config.image_channels,
        config.image_size,
        config.image_size).to(config.device)

    # images is a list of num_timesteps images batches, e.g. List[Tensor(NCHW)]
    images = pipeline.sampling(model, noisy_sample, device=config.device, save_all_steps=True)

    fig = plt.figure()
    ims = []
    for i in range(pipeline.num_timesteps):
        img = postprocess(images[i][0].unsqueeze(0))
        img = Image.fromarray(img[0])
        im = plt.imshow(img, animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=5000)
    path_to_save_animation = Path(config.output_dir, "samples", "diffusion.gif")
    animate.save(str(path_to_save_animation))

