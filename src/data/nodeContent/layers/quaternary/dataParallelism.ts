import { NodeContent } from '../../../../types/content';

export const dataParallelismContent: NodeContent = {
  title: 'Data Parallelism',
  description: 'A distributed training strategy where the same model is replicated across multiple devices, with each device processing different batches of data.',
  concepts: [
    'Batch Distribution',
    'Gradient Synchronization',
    'All-Reduce Operations',
    'Scaling Efficiency',
    'Batch Size Scaling'
  ],
  examples: [
    {
      language: 'python',
      description: 'Data parallel training with PyTorch',
      code: `import torch.nn as nn
from torch.nn.parallel import DataParallel

# Wrap model with DataParallel
model = YourModel()
if torch.cuda.device_count() > 1:
    model = DataParallel(model)
model = model.to('cuda')

# Training loop remains the same
for batch in dataloader:
    inputs, labels = batch
    inputs = inputs.to('cuda')
    labels = labels.to('cuda')
    
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()`
    }
  ],
  resources: [
    {
      title: 'Data Parallelism Guide',
      description: 'PyTorch data parallel training guide',
      url: 'https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html'
    },
    {
      title: 'Distributed Training',
      description: 'Best practices for data parallel training',
      url: 'https://pytorch.org/docs/stable/distributed.html'
    }
  ],
  prerequisites: ['Deep Learning', 'Distributed Computing', 'GPU Programming'],
  relatedTopics: ['Distributed Training', 'Model Parallelism', 'Batch Processing']
};