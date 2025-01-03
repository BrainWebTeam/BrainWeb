import { NodeContent } from '../../../../types/content';

export const lossFunctionsContent: NodeContent = {
  title: 'Loss Functions',
  description: 'Mathematical functions that measure the difference between predicted and actual values, guiding the learning process of machine learning models.',
  concepts: [
    'Cross-Entropy Loss',
    'Mean Squared Error',
    'Hinge Loss',
    'Focal Loss',
    'Custom Loss Design'
  ],
  examples: [
    {
      language: 'python',
      description: 'Common loss functions implementation',
      code: `import torch
import torch.nn.functional as F

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()

# Different loss functions
mse_loss = F.mse_loss(pred, target)
ce_loss = F.cross_entropy(pred, target)
kl_loss = F.kl_div(pred.log_softmax(dim=-1), target)`
    }
  ],
  resources: [
    {
      title: 'Loss Functions Guide',
      description: 'Overview of common loss functions',
      url: 'https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html'
    },
    {
      title: 'Focal Loss Paper',
      description: 'Paper introducing focal loss',
      url: 'https://arxiv.org/abs/1708.02002'
    }
  ],
  prerequisites: ['Calculus', 'Machine Learning Basics', 'Optimization'],
  relatedTopics: ['Optimization', 'Model Training', 'Gradient Descent']
};