import { NodeContent } from '../../../../types/content';

export const mixedPrecisionContent: NodeContent = {
  title: 'Mixed Precision Training',
  description: 'A technique that combines different numerical precisions during training to reduce memory usage and increase training speed while maintaining model accuracy.',
  concepts: [
    'FP16/FP32 Training',
    'Loss Scaling',
    'Numerical Stability',
    'Memory Optimization',
    'Hardware Acceleration'
  ],
  examples: [
    {
      language: 'python',
      description: 'Mixed precision training with PyTorch',
      code: `from torch.cuda.amp import autocast, GradScaler

# Initialize gradient scaler
scaler = GradScaler()

# Training loop with mixed precision
for inputs, labels in dataloader:
    optimizer.zero_grad()
    
    # Automatic mixed precision
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    # Scale gradients and optimize
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()`
    }
  ],
  resources: [
    {
      title: 'Mixed Precision Guide',
      description: 'NVIDIA guide to mixed precision training',
      url: 'https://developer.nvidia.com/automatic-mixed-precision'
    },
    {
      title: 'PyTorch AMP',
      description: 'PyTorch automatic mixed precision docs',
      url: 'https://pytorch.org/docs/stable/amp.html'
    }
  ],
  prerequisites: ['Deep Learning', 'GPU Computing', 'Numerical Computing'],
  relatedTopics: ['Model Optimization', 'Training Speed', 'Memory Management']
};