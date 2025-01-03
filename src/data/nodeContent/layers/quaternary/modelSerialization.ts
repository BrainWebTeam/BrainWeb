import { NodeContent } from '../../../../types/content';

export const modelSerializationContent: NodeContent = {
  title: 'Model Serialization',
  description: 'Techniques for saving and loading machine learning models, enabling model sharing and deployment.',
  concepts: [
    'Model Format Standards',
    'State Dictionary Management',
    'Version Compatibility',
    'Optimization for Deployment',
    'Cross-Platform Support'
  ],
  examples: [
    {
      language: 'python',
      description: 'Model serialization techniques',
      code: `import torch
import onnx
import torch.onnx

# Save/Load PyTorch Model
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)`
    }
  ],
  resources: [
    {
      title: 'PyTorch Model Saving',
      description: 'Guide to saving and loading models',
      url: 'https://pytorch.org/tutorials/beginner/saving_loading_models.html'
    },
    {
      title: 'ONNX Format',
      description: 'Open Neural Network Exchange format',
      url: 'https://onnx.ai/get-started'
    }
  ],
  prerequisites: ['Deep Learning', 'File I/O', 'Model Architecture'],
  relatedTopics: ['Model Deployment', 'Checkpointing', 'Model Formats']
};