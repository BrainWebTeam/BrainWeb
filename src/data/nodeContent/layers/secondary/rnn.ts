import { NodeContent } from '../../../../types/content';

export const rnnContent: NodeContent = {
  title: 'Recurrent Neural Networks',
  description: 'Neural networks designed to work with sequential data by maintaining an internal state (memory) that captures information about previous inputs.',
  concepts: [
    'Hidden State',
    'Backpropagation Through Time',
    'Long Short-Term Memory (LSTM)',
    'Gated Recurrent Units (GRU)',
    'Sequence Processing'
  ],
  examples: [
    {
      language: 'python',
      description: 'LSTM implementation with PyTorch',
      code: `import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions`
    }
  ],
  resources: [
    {
      title: 'Understanding LSTM Networks',
      description: 'Comprehensive LSTM tutorial',
      url: 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/'
    },
    {
      title: 'RNN Tutorial',
      description: 'Step-by-step RNN implementation guide',
      url: 'https://www.tensorflow.org/guide/keras/rnn'
    }
  ],
  prerequisites: ['Neural Networks', 'Backpropagation', 'Sequential Data Processing'],
  relatedTopics: ['LSTM', 'GRU', 'Sequence Modeling']
};