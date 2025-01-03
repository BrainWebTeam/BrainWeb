import { NodeContent } from '../../../../types/content';

export const pruningContent: NodeContent = {
  title: 'Model Pruning',
  description: 'A technique to reduce model size and complexity by removing unnecessary weights or neurons while maintaining performance.',
  concepts: [
    'Weight Pruning',
    'Channel Pruning',
    'Structured Pruning',
    'Magnitude-based Pruning',
    'Iterative Pruning'
  ],
  examples: [
    {
      language: 'python',
      description: 'PyTorch pruning implementation',
      code: `import torch.nn.utils.prune as prune

# Apply L1 unstructured pruning
prune.l1_unstructured(
    module=model.conv1,
    name='weight',
    amount=0.3  # Prune 30% of weights
)

# Apply structured pruning
prune.ln_structured(
    module=model.conv1,
    name='weight',
    amount=0.5,
    n=2,
    dim=0  # Prune output channels
)

# Make pruning permanent
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')`
    }
  ],
  resources: [
    {
      title: 'Neural Network Pruning',
      description: 'Comprehensive guide to model pruning',
      url: 'https://arxiv.org/abs/1506.02626'
    },
    {
      title: 'PyTorch Pruning Tutorial',
      description: 'Official pruning documentation',
      url: 'https://pytorch.org/tutorials/intermediate/pruning_tutorial.html'
    }
  ],
  prerequisites: ['Neural Networks', 'Model Compression', 'Network Architecture'],
  relatedTopics: ['Model Compression', 'Quantization', 'Network Architecture']
};