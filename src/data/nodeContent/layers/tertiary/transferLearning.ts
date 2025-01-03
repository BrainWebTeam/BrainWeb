import { NodeContent } from '../../../../types/content';

export const transferLearningContent: NodeContent = {
  title: 'Transfer Learning',
  description: 'A machine learning technique where a model developed for one task is reused as the starting point for a model on a second task, saving time and resources.',
  concepts: [
    'Pre-trained Models',
    'Fine-tuning',
    'Feature Extraction',
    'Domain Adaptation',
    'Knowledge Transfer'
  ],
  examples: [
    {
      language: 'python',
      description: 'Transfer learning with pre-trained ResNet',
      code: `import torch
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Modify final layer for new task
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Train only the final layer
optimizer = torch.optim.Adam(model.fc.parameters())`
    }
  ],
  resources: [
    {
      title: 'Transfer Learning Guide',
      description: 'Comprehensive guide to transfer learning',
      url: 'https://cs231n.github.io/transfer-learning/'
    },
    {
      title: 'PyTorch Transfer Learning',
      description: 'Official PyTorch transfer learning tutorials',
      url: 'https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html'
    }
  ],
  prerequisites: ['Deep Learning', 'Neural Networks', 'Model Training'],
  relatedTopics: ['Fine-tuning', 'Pre-trained Models', 'Domain Adaptation']
};