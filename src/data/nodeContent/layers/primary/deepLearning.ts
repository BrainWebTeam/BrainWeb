import { NodeContent } from '../../../../types/content';

export const deepLearningContent: NodeContent = {
  title: 'Deep Learning',
  description: 'A subset of machine learning based on artificial neural networks with multiple layers that progressively extract higher-level features from raw input.',
  concepts: [
    'Neural Network Architecture',
    'Backpropagation',
    'Activation Functions',
    'Loss Functions',
    'Gradient Descent'
  ],
  examples: [
    {
      language: 'python',
      description: 'Deep neural network with PyTorch',
      code: `import torch
import torch.nn as nn

class DeepNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)`
    }
  ],
  resources: [
    {
      title: 'Deep Learning Book',
      description: 'Comprehensive deep learning textbook',
      url: 'https://www.deeplearningbook.org/'
    },
    {
      title: 'Deep Learning Specialization',
      description: 'Andrew Ng\'s deep learning courses',
      url: 'https://www.coursera.org/specializations/deep-learning'
    }
  ],
  prerequisites: ['Machine Learning', 'Linear Algebra', 'Calculus'],
  relatedTopics: ['Neural Networks', 'Computer Vision', 'Natural Language Processing']
};