import { NodeContent } from '../../types/content';

export const deepLearningContent: NodeContent = {
  title: 'Deep Learning',
  description: 'Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep neural networks) to progressively extract higher-level features from raw input.',
  concepts: [
    'Neural Network Architecture',
    'Backpropagation',
    'Activation Functions',
    'Loss Functions',
    'Gradient Descent Optimization'
  ],
  examples: [
    {
      language: 'python',
      description: 'Creating a simple neural network with PyTorch',
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
        return self.layers(x)

# Create model
model = SimpleNN()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())`
    }
  ],
  resources: [
    {
      title: 'Deep Learning Book',
      description: 'Comprehensive deep learning textbook by Ian Goodfellow et al.',
      url: 'https://www.deeplearningbook.org/'
    },
    {
      title: 'PyTorch Tutorials',
      description: 'Official PyTorch tutorials and examples',
      url: 'https://pytorch.org/tutorials/'
    }
  ],
  prerequisites: ['Machine Learning Basics', 'Linear Algebra', 'Calculus'],
  relatedTopics: ['Computer Vision', 'Natural Language Processing', 'Reinforcement Learning']
};