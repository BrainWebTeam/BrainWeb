import { NodeContent } from '../../../../types/content';

export const quantizationContent: NodeContent = {
  title: 'Model Quantization',
  description: 'A technique to reduce model size and inference time by converting floating-point weights to lower-precision formats.',
  concepts: [
    'Post-Training Quantization',
    'Quantization-Aware Training',
    'Dynamic Range Quantization',
    'Fixed-Point Arithmetic',
    'Precision Calibration'
  ],
  examples: [
    {
      language: 'python',
      description: 'PyTorch quantization example',
      code: `import torch

# Prepare model for quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibrate with sample data
model(calibration_data)

# Convert to quantized model
torch.quantization.convert(model, inplace=True)

# Compare model sizes
original_size = os.path.getsize('fp32_model.pth')
quantized_size = os.path.getsize('int8_model.pth')
compression_ratio = original_size / quantized_size`
    }
  ],
  resources: [
    {
      title: 'PyTorch Quantization',
      description: 'Official quantization tutorial',
      url: 'https://pytorch.org/docs/stable/quantization.html'
    },
    {
      title: 'Quantization Paper',
      description: 'Deep learning model compression survey',
      url: 'https://arxiv.org/abs/1710.09282'
    }
  ],
  prerequisites: ['Deep Learning', 'Model Optimization', 'Computer Architecture'],
  relatedTopics: ['Model Compression', 'Pruning', 'Efficient Inference']
};