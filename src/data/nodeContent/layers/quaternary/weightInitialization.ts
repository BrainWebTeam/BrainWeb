import { NodeContent } from '../../../../types/content';

export const weightInitializationContent: NodeContent = {
  title: 'Weight Initialization',
  description: 'Techniques for setting initial values of neural network parameters to enable stable and efficient training.',
  concepts: [
    'Xavier/Glorot Initialization',
    'He Initialization',
    'Orthogonal Initialization',
    'Scale Factors',
    'Variance Analysis'
  ],
  examples: [
    {
      language: 'python',
      description: 'Different weight initialization methods',
      code: `import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # He initialization
        nn.init.kaiming_normal_(
            m.weight, 
            mode='fan_out',
            nonlinearity='relu'
        )

# Apply initialization
model.apply(init_weights)`
    }
  ],
  resources: [
    {
      title: 'Weight Initialization Guide',
      description: 'Deep dive into initialization techniques',
      url: 'https://pytorch.org/docs/stable/nn.init.html'
    },
    {
      title: 'Xavier Initialization Paper',
      description: 'Original Xavier initialization paper',
      url: 'http://proceedings.mlr.press/v9/glorot10a.html'
    }
  ],
  prerequisites: ['Neural Networks', 'Linear Algebra', 'Optimization'],
  relatedTopics: ['Model Training', 'Network Architecture', 'Gradient Flow']
};