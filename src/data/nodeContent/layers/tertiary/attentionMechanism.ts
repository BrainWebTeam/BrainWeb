import { NodeContent } from '../../../../types/content';

export const attentionMechanismContent: NodeContent = {
  title: 'Attention Mechanism',
  description: 'A neural network component that allows models to focus on specific parts of the input when processing sequential data.',
  concepts: [
    'Self-Attention',
    'Multi-Head Attention',
    'Query-Key-Value Framework',
    'Attention Weights',
    'Position Encoding'
  ],
  examples: [
    {
      language: 'python',
      description: 'Simple attention mechanism implementation',
      code: `import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        
    def forward(self, encoder_outputs, decoder_hidden):
        # Calculate attention scores
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat decoder hidden state
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.attention(
            torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)
        
        # Calculate attention weights
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention_weights = F.softmax(
            torch.bmm(v, energy).squeeze(1), dim=1)
        
        # Apply attention weights
        context = torch.bmm(attention_weights.unsqueeze(1),
                          encoder_outputs)
        
        return context, attention_weights`
    }
  ],
  resources: [
    {
      title: 'Attention Explained',
      description: 'Visual guide to attention mechanisms',
      url: 'https://jalammar.github.io/illustrated-transformer/'
    },
    {
      title: 'Attention Paper',
      description: 'Original attention mechanism paper',
      url: 'https://arxiv.org/abs/1409.0473'
    }
  ],
  prerequisites: ['Neural Networks', 'Deep Learning', 'Linear Algebra'],
  relatedTopics: ['Transformers', 'Sequence Models', 'Natural Language Processing']
};