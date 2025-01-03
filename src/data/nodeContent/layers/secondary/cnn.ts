import { NodeContent } from '../../../../types/content';

export const cnnContent: NodeContent = {
  title: 'Convolutional Neural Networks',
  description: 'Deep learning architecture specifically designed for processing grid-like data, particularly effective for image and video analysis.',
  concepts: [
    'Convolutional Layers',
    'Pooling Operations',
    'Feature Maps',
    'Receptive Fields',
    'Transfer Learning'
  ],
  examples: [
    {
      language: 'python',
      description: 'Simple CNN with PyTorch',
      code: `import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )`
    }
  ],
  resources: [
    {
      title: 'CS231n CNN Course',
      description: 'Stanford\'s CNN for Visual Recognition course',
      url: 'http://cs231n.stanford.edu/'
    },
    {
      title: 'CNN Explainer',
      description: 'Interactive visualization of CNNs',
      url: 'https://poloclub.github.io/cnn-explainer/'
    }
  ],
  prerequisites: ['Neural Networks', 'Linear Algebra', 'Computer Vision Basics'],
  relatedTopics: ['Computer Vision', 'Image Classification', 'Object Detection']
};