import torch
import numpy as np


class DDPMPipeline:
    def __init__(self, beta_start=1e-4, beta_end=1e-2, num_timesteps=1000):
        # Betas settings are in section 4 of https://arxiv.org/pdf/2006.11239.pdf
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas

        # alpha-hat it the paper
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

    def _broadcast(self, values, broadcast_to):
        values = values.flatten()

        while len(values.shape) < len(broadcast_to.shape):
            values = values.unsqueeze(-1)

        return values

    def forward_diffusion(self, images, timesteps):
        """
        https://arxiv.org/pdf/2006.11239.pdf, equation (14), the term inside epsilon_theta
        :return:
        """
        gaussian_noise = torch.randn(images.shape).to(images.device)
        alpha_hat = self.alphas_hat[timesteps].to(images.device)
        alpha_hat = self._broadcast(alpha_hat, images)

        return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * gaussian_noise, gaussian_noise


