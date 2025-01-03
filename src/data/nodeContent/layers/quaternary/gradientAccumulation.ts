import { NodeContent } from '../../../../types/content';

export const gradientAccumulationContent: NodeContent = {
  title: 'Gradient Accumulation',
  description: 'A technique that enables training with larger effective batch sizes by accumulating gradients over multiple forward and backward passes before updating model parameters.',
  concepts: [
    'Virtual Batch Size',
    'Memory Optimization',
    'Update Frequency',
    'Gradient Scaling',
    'Batch Normalization'
  ],
  examples: [
    {
      language: 'python',
      description: 'Gradient accumulation implementation',
      code: `# Training with gradient accumulation
accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(dataloader):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Scale loss to account for accumulation
    loss = loss / accumulation_steps
    loss.backward()
    
    # Update weights after accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()`
    }
  ],
  resources: [
    {
      title: 'Gradient Accumulation Guide',
      description: 'Best practices for gradient accumulation',
      url: 'https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation'
    },
    {
      title: 'Large Batch Training',
      description: 'Training with large batch sizes',
      url: 'https://arxiv.org/abs/1706.02677'
    }
  ],
  prerequisites: ['Deep Learning', 'Optimization', 'Memory Management'],
  relatedTopics: ['Batch Size', 'Memory Optimization', 'Training Techniques']
};