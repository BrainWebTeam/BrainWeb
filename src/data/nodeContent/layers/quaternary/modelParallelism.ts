import { NodeContent } from '../../../../types/content';

export const modelParallelismContent: NodeContent = {
  title: 'Model Parallelism',
  description: 'A distributed training approach where different parts of a model are placed on different devices, enabling training of very large models.',
  concepts: [
    'Layer Distribution',
    'Pipeline Parallelism',
    'Memory Management',
    'Device Communication',
    'Load Balancing'
  ],
  examples: [
    {
      language: 'python',
      description: 'Model parallel implementation',
      code: `import torch.nn as nn

class ModelParallelResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # First part goes to GPU 0
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        ).to('cuda:0')
        
        # Second part goes to GPU 1
        self.seq2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ).to('cuda:1')
    
    def forward(self, x):
        x = x.to('cuda:0')
        x = self.seq1(x)
        x = x.to('cuda:1')
        x = self.seq2(x)
        return x`
    }
  ],
  resources: [
    {
      title: 'Model Parallelism Guide',
      description: 'Deep dive into model parallel training',
      url: 'https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html'
    },
    {
      title: 'Pipeline Parallelism',
      description: 'Advanced pipeline parallel techniques',
      url: 'https://arxiv.org/abs/1811.06965'
    }
  ],
  prerequisites: ['Distributed Computing', 'GPU Programming', 'Deep Learning'],
  relatedTopics: ['Pipeline Parallelism', 'Distributed Training', 'Large Models']
};