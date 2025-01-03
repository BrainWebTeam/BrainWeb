import { NodeContent } from '../../../../types/content';

export const dropoutContent: NodeContent = {
  title: 'Dropout',
  description: 'A regularization technique that randomly deactivates neurons during training to prevent overfitting and improve generalization.',
  concepts: [
    'Dropout Rate',
    'Training vs Inference',
    'Monte Carlo Dropout',
    'Spatial Dropout',
    'Adaptive Dropout'
  ],
  examples: [
    {
      language: 'python',
      description: 'Implementing dropout in a neural network',
      code: `import torch.nn as nn

class DropoutMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)`
    }
  ],
  resources: [
    {
      title: 'Dropout Paper',
      description: 'Original dropout paper',
      url: 'https://jmlr.org/papers/v15/srivastava14a.html'
    },
    {
      title: 'Dropout Guide',
      description: 'Comprehensive guide to dropout',
      url: 'https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html'
    }
  ],
  prerequisites: ['Neural Networks', 'Regularization', 'Model Training'],
  relatedTopics: ['Regularization', 'Overfitting', 'Model Architecture']
};