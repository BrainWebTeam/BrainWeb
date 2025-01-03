import { NodeContent } from '../../../../types/content';

export const modelCompressionContent: NodeContent = {
  title: 'Model Compression',
  description: 'Techniques to reduce the size and computational requirements of deep learning models while maintaining performance.',
  concepts: [
    'Quantization',
    'Pruning',
    'Knowledge Distillation',
    'Weight Sharing',
    'Model Architecture Optimization'
  ],
  examples: [
    {
      language: 'python',
      description: 'Post-training quantization',
      code: `import torch

# Define quantization configuration
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# Compare model sizes
original_size = os.path.getsize('model.pth')
quantized_size = os.path.getsize('quantized_model.pth')

compression_ratio = original_size / quantized_size`
    }
  ],
  resources: [
    {
      title: 'Model Compression Survey',
      description: 'Comprehensive overview of compression techniques',
      url: 'https://arxiv.org/abs/1710.09282'
    },
    {
      title: 'TensorFlow Model Optimization',
      description: 'Official TensorFlow model optimization toolkit',
      url: 'https://www.tensorflow.org/model_optimization'
    }
  ],
  prerequisites: ['Deep Learning', 'Neural Networks', 'Model Architecture'],
  relatedTopics: ['Quantization', 'Pruning', 'Knowledge Distillation']
};