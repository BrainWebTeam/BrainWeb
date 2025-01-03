import { NodeContent } from '../../../../types/content';

export const multiTaskLearningContent: NodeContent = {
  title: 'Multi-Task Learning',
  description: 'A training approach where a single model learns to perform multiple related tasks simultaneously, leveraging shared representations and knowledge transfer between tasks.',
  concepts: [
    'Task Sharing',
    'Hard/Soft Parameter Sharing',
    'Task Weighting',
    'Gradient Balancing',
    'Cross-task Transfer'
  ],
  examples: [
    {
      language: 'python',
      description: 'Multi-task neural network implementation',
      code: `import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_per_task):
        super().__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Linear(128, num_classes)
            for num_classes in num_classes_per_task
        ])
    
    def forward(self, x):
        shared_features = self.shared(x)
        return [head(shared_features) for head in self.task_heads]`
    }
  ],
  resources: [
    {
      title: 'Multi-Task Learning Survey',
      description: 'Comprehensive overview of MTL techniques',
      url: 'https://arxiv.org/abs/1706.05098'
    },
    {
      title: 'MTL in Practice',
      description: 'Practical guide to multi-task learning',
      url: 'https://ruder.io/multi-task-learning-nlp/'
    }
  ],
  prerequisites: ['Deep Learning', 'Transfer Learning', 'Model Architecture'],
  relatedTopics: ['Transfer Learning', 'Parameter Sharing', 'Task Learning']
};