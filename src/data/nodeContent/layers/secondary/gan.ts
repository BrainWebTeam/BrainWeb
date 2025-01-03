import { NodeContent } from '../../../../types/content';

export const ganContent: NodeContent = {
  title: 'Generative Adversarial Networks',
  description: 'A framework where two neural networks compete against each other to generate realistic synthetic data.',
  concepts: [
    'Generator Network',
    'Discriminator Network',
    'Adversarial Training',
    'Mode Collapse',
    'Wasserstein GAN'
  ],
  examples: [
    {
      language: 'python',
      description: 'Simple GAN implementation',
      code: `import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )`
    }
  ],
  resources: [
    {
      title: 'GAN Lab',
      description: 'Interactive visualization of GANs',
      url: 'https://poloclub.github.io/ganlab/'
    },
    {
      title: 'GAN Tutorial',
      description: 'NIPS 2016 Tutorial on GANs',
      url: 'https://arxiv.org/abs/1701.00160'
    }
  ],
  prerequisites: ['Deep Learning', 'Probability Theory', 'Neural Networks'],
  relatedTopics: ['DCGAN', 'StyleGAN', 'Image Generation']
};