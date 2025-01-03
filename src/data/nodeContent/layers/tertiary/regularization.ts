import { NodeContent } from '../../../../types/content';

export const regularizationContent: NodeContent = {
  title: 'Regularization',
  description: 'Techniques to prevent overfitting and improve model generalization by adding constraints or penalties to the learning process.',
  concepts: [
    'L1/L2 Regularization',
    'Dropout',
    'Batch Normalization',
    'Early Stopping',
    'Data Augmentation'
  ],
  examples: [
    {
      language: 'python',
      description: 'Different regularization techniques',
      code: `import torch.nn as nn

class RegularizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# L2 regularization in optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 penalty
)`
    }
  ],
  resources: [
    {
      title: 'Regularization Guide',
      description: 'Comprehensive guide to regularization techniques',
      url: 'https://www.deeplearningbook.org/contents/regularization.html'
    },
    {
      title: 'Dropout Paper',
      description: 'Original dropout paper',
      url: 'https://jmlr.org/papers/v15/srivastava14a.html'
    }
  ],
  prerequisites: ['Neural Networks', 'Model Training', 'Optimization'],
  relatedTopics: ['Overfitting', 'Model Complexity', 'Training Techniques']
};