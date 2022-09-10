# diffusion-DDPM
PyTorch Implementation of "Denoising Diffusion Probabilistic Models", Ho et al., 2020
<p align="center">
  <img src="https://user-images.githubusercontent.com/8377365/188951630-d3e38fb0-9545-4208-bf3b-2296bde10864.png" />
</p>

## Overview
This repo is yet another denoising diffusion probabilistic model (DDPM) implementation. This repo tries to stick to the original paper as close as possible.

The [straightforward UNet model definition](https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/unet.py) (without any fancy model builders, helpers, etc.) was specifically intentional because it can be quite difficult sometimes to get and understand the original model architecture behind all the abstraction layers and blocks and see the underlying entities clearly.
However some kind of automated model generation with configuration files is handy while experimenting, hence will be added in the nearest future. 

Some equations are borrowed from [this](https://lilianweng.github.io/posts/2021-07-11-diffusion-models) blog post which demystifies whole math behind the diffusion process.

## Diffusion process
Diffusion process was implemented as a part of a class called [DDPMPipeline](https://github.com/mattroz/diffusion-ddpm/blob/main/src/scheduler/ddpm.py#L9), which containes forward $q(x_t \vert x_{t-1})$ and backward $p_\theta(x_{t-1} \vert x_t)$ diffusion processes.

Backward diffusion process [applies Gaussian noise](https://github.com/mattroz/diffusion-ddpm/blob/main/src/scheduler/ddpm.py#L21) to the input image in a scheduleded manner. 
Forward diffusion process is a process which "denoises" an image using model predictions. It is worth to mention, that UNet model in this particular process predicts some kind of noise residual, and the final "denoised" image is obtained by [applying the following equation](https://github.com/mattroz/diffusion-ddpm/blob/main/src/scheduler/ddpm.py#L67): 
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon(x_t,t)) + \sigma_tz$$

Here, $\epsilon$ is the UNet model, $\alpha_t$, $\bar{\alpha}_t$ [are precomputed](https://github.com/mattroz/diffusion-ddpm/blob/main/src/scheduler/ddpm.py#L13) and $\sigma_t$ is [calculated](https://github.com/mattroz/diffusion-ddpm/blob/main/src/scheduler/ddpm.py#L65) using these precomputed values at forward diffusion step.

<p align="center">
  <img src="https://user-images.githubusercontent.com/8377365/188951361-0168a56b-38fd-4048-8351-de9b3a601299.png" />
</p>

## UNet

As stated in the original paper:
> * Our neural network architecture follows the backbone of PixelCNN++, which is a U-Net based on a Wide ResNet. 
> * We replaced weight normalization with [group normalization](https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/layers.py#L36) to make the implementation simpler. 
> * Our 32×32 models use four feature map resolutions (32×32 to 4×4), and our 256×256 models use six.  
> * All models have two [convolutional residual blocks](https://github.com/mattroz/diffusion-ddpm/blob/00de3c830f3765a347fc5efe2e3fc21d6f597104/src/model/layers.py#L305) per resolution level and [self-attention blocks](https://github.com/mattroz/diffusion-ddpm/blob/00de3c830f3765a347fc5efe2e3fc21d6f597104/src/model/layers.py#L124) at the 16×16 resolution between the convolutional blocks. 
> * Diffusion time is specified by adding the [Transformer sinusoidal position embedding](https://github.com/mattroz/diffusion-ddpm/blob/00de3c830f3765a347fc5efe2e3fc21d6f597104/src/model/layers.py#L6) into each residual block.

This implementation follows default ResNet blocks architecture without any multiplying factors for simplicity. Also current UNet implementation works better with 128×128 resolution (see next sections) and thus has 5 feature map resoltuions (128 &rarr; 64 &rarr; 32 &rarr; 16 &rarr; 8).
It is worth noting that subsequent papers suggests more appropriate and better UNet architectures for the diffusion problem.

## Results

Training was performed on two datasets:
* [smithsonian-butterflies-subset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) by HuggingFace
* [croupier-mtg-dataset](https://huggingface.co/datasets/alkzar90/croupier-mtg-dataset) by [alcazar90](https://github.com/alcazar90/croupier-mtg-dataset)

### 128×128 resolution
All 128×128 models were trained for 300 epochs with cosine annealing with initial learning rate set to 2e-4, batch size 6 and 1000 diffusion timesteps. 
#### Training on smithsonian-butterflies-subset
300 epochs, 50266 steps
Epoch 4             |  Epoch 99
:-------------------------:|:-------------------------:
![0004](https://user-images.githubusercontent.com/8377365/189268993-97f7b8be-4ab0-4cc9-af46-87582bfad1b4.png)  |  ![0099](https://user-images.githubusercontent.com/8377365/189269009-49ee9a26-7c63-4bdf-b4d1-79d1e034cc12.png)
Epoch 204             |  Epoch 300
![0204](https://user-images.githubusercontent.com/8377365/189269020-c37756b6-0518-4b90-8d28-64d40eaedc0e.png)  |   ![0300](https://user-images.githubusercontent.com/8377365/189269042-19ac4e36-92c7-4141-b43a-d1b405ad108e.png)
Sampling from the epoch=300 | Sampling from the epoch=300
![diffusion1](https://user-images.githubusercontent.com/8377365/189269282-82b13b7b-eb6b-4746-8a63-6c6b06f40ebc.gif)  |   ![diffusion2](https://user-images.githubusercontent.com/8377365/189269404-59f205c5-95ff-4b4e-9447-e68409f61f9e.gif)

#### Training on croupier-mtg-dataset
300 epoch, 72599 steps
Epoch 4             |  Epoch 99
:-------------------------:|:-------------------------:
![0004](https://user-images.githubusercontent.com/8377365/189183793-c3da77ab-f306-4a94-bd5e-df500bfe3465.png)  |  ![0099](https://user-images.githubusercontent.com/8377365/189183825-37028de4-030b-4471-88e8-2d17094cec8a.png)
Epoch 204             |  Epoch 300
![0204](https://user-images.githubusercontent.com/8377365/189183859-d70a572f-1027-4af5-948b-057c042ab508.png)  |  ![0300](https://user-images.githubusercontent.com/8377365/189183877-63a705da-1489-497f-9d8a-c8be9bdf0bdf.png)
Sampling from the epoch=300 | Sampling from the epoch=300
![diffusion1](https://user-images.githubusercontent.com/8377365/189268712-2cb1fd0c-b566-4058-893b-cbba2d949eb2.gif)   |   ![diffusion2](https://user-images.githubusercontent.com/8377365/189268713-da20e5fc-9ce5-45ce-977c-d6e1db35e090.gif)

### 256×256 resolution
All 256×256 models were trained for 300 epochs with cosine annealing with initial learning rate set to 2e-5, batch size 6 and 1000 diffusion timesteps.
#### Training on smithsonian-butterflies-subset
300 epochs, 50266 steps
Epoch 4             |  Epoch 100
:-------------------------:|:-------------------------:
![0004](https://user-images.githubusercontent.com/8377365/189496165-84d677b0-8b13-4eb1-a6d6-09879db11fc1.png)  |  ![0100](https://user-images.githubusercontent.com/8377365/189496166-286aedb5-7b5e-4317-9cab-cd0cf94487b3.png)
Epoch 205           |  Epoch 300
![0205](https://user-images.githubusercontent.com/8377365/189496168-58933a78-f276-4d6b-8ee3-8a5d94db7b9d.png)  |  ![0300](https://user-images.githubusercontent.com/8377365/189496174-2bfc2d75-6e0e-493b-ab8f-215c25a5175a.png)

[TODO description]
