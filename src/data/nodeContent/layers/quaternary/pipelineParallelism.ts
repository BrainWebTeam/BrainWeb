import { NodeContent } from '../../../../types/content';

export const pipelineParallelismContent: NodeContent = {
  title: 'Pipeline Parallelism',
  description: 'A distributed training approach that splits model layers across devices and processes mini-batches in a pipelined fashion to maximize hardware utilization.',
  concepts: [
    'Micro-batching',
    'Pipeline Scheduling',
    'Bubble Overhead',
    'Memory Efficiency',
    'Pipeline Stages'
  ],
  examples: [
    {
      language: 'python',
      description: 'Pipeline parallel implementation',
      code: `import torch
from torch.distributed.pipeline.sync import Pipe

class PipelineModel(nn.Module):
    def __init__(self, num_stages):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            ) for _ in range(num_stages)
        ])
        
        # Wrap model in Pipe
        self.model = Pipe(
            self.stages,
            chunks=8  # Number of micro-batches
        )
    
    def forward(self, x):
        return self.model(x)`
    }
  ],
  resources: [
    {
      title: 'GPipe Paper',
      description: 'Original pipeline parallelism paper',
      url: 'https://arxiv.org/abs/1811.06965'
    },
    {
      title: 'Pipeline Tutorial',
      description: 'Guide to pipeline parallel training',
      url: 'https://pytorch.org/docs/stable/pipeline.html'
    }
  ],
  prerequisites: ['Model Parallelism', 'Distributed Computing', 'Deep Learning'],
  relatedTopics: ['Model Parallelism', 'Distributed Training', 'Memory Optimization']
};