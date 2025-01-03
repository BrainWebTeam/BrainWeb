import { NodeContent } from '../../../types/content';

export const neuralNetworksContent: NodeContent = {
  title: 'Neural Networks',
  description: 'Computational models inspired by biological neural networks, forming the foundation of deep learning and modern AI systems.',
  concepts: [
    'Neurons and Layers',
    'Activation Functions',
    'Backpropagation',
    'Weight Initialization',
    'Network Architectures'
  ],
  examples: [
    {
      language: 'python',
      description: 'Simple neural network with PyTorch',
      code: `import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)`
    }
  ],
  resources: [
    {
      title: '3Blue1Brown Neural Networks',
      description: 'Visual introduction to neural networks',
      url: 'https://www.3blue1brown.com/topics/neural-networks'
    },
    {
      title: 'Neural Networks and Deep Learning',
      description: 'Free online book by Michael Nielsen',
      url: 'http://neuralnetworksanddeeplearning.com/'
    }
  ],
  prerequisites: ['Linear Algebra', 'Calculus', 'Programming Basics'],
  relatedTopics: ['Deep Learning', 'Backpropagation', 'Activation Functions']
};