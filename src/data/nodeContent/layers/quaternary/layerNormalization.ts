import { NodeContent } from '../../../../types/content';

export const layerNormalizationContent: NodeContent = {
  title: 'Layer Normalization',
  description: 'A normalization technique that normalizes the inputs across the features, making it independent of batch size and particularly effective for recurrent neural networks.',
  concepts: [
    'Feature Statistics',
    'Scale and Shift Parameters',
    'Sequence Normalization',
    'Batch Independence',
    'Transformer Applications'
  ],
  examples: [
    {
      language: 'python',
      description: 'Layer normalization implementation',
      code: `import torch.nn as nn

class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, hidden=None):
        output, (h_n, c_n) = self.lstm(x, hidden)
        normalized_output = self.layer_norm(output)
        return normalized_output, (h_n, c_n)`
    }
  ],
  resources: [
    {
      title: 'Layer Normalization Paper',
      description: 'Original layer normalization paper',
      url: 'https://arxiv.org/abs/1607.06450'
    },
    {
      title: 'Layer Norm Guide',
      description: 'Deep dive into layer normalization',
      url: 'https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html'
    }
  ],
  prerequisites: ['Neural Networks', 'Deep Learning', 'Batch Normalization'],
  relatedTopics: ['Batch Normalization', 'Instance Normalization', 'Transformer Architecture']
};