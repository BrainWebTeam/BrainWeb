import { NodeContent } from '../../../../types/content';

export const batchNormalizationContent: NodeContent = {
  title: 'Batch Normalization',
  description: 'A technique to normalize the intermediate activations of neural networks, improving training stability and speed.',
  concepts: [
    'Normalization Statistics',
    'Running Averages',
    'Internal Covariate Shift',
    'Training vs Inference',
    'Batch Dependencies'
  ],
  examples: [
    {
      language: 'python',
      description: 'Implementing batch normalization',
      code: `import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))`
    }
  ],
  resources: [
    {
      title: 'Batch Normalization Paper',
      description: 'Original batch normalization paper',
      url: 'https://arxiv.org/abs/1502.03167'
    },
    {
      title: 'Understanding BatchNorm',
      description: 'Deep dive into batch normalization',
      url: 'https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html'
    }
  ],
  prerequisites: ['Neural Networks', 'Deep Learning', 'Optimization'],
  relatedTopics: ['Layer Normalization', 'Model Training', 'Neural Architecture']
};