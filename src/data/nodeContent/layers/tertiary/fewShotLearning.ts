import { NodeContent } from '../../../../types/content';

export const fewShotLearningContent: NodeContent = {
  title: 'Few-Shot Learning',
  description: 'Machine learning techniques that enable models to learn from very few examples, similar to human learning ability.',
  concepts: [
    'Meta-Learning',
    'Prototypical Networks',
    'Matching Networks',
    'MAML (Model-Agnostic Meta-Learning)',
    'Siamese Networks'
  ],
  examples: [
    {
      language: 'python',
      description: 'Prototypical network implementation',
      code: `import torch
import torch.nn as nn

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, support_set, query_set, n_way):
        # Encode support and query sets
        z_support = self.encoder(support_set)
        z_query = self.encoder(query_set)
        
        # Calculate prototypes
        z_proto = z_support.reshape(n_way, -1, z_support.shape[-1]).mean(1)
        
        # Calculate distances
        dists = torch.cdist(z_query, z_proto)
        
        return -dists  # Return negative distances as logits`
    }
  ],
  resources: [
    {
      title: 'Few-Shot Learning Survey',
      description: 'Comprehensive overview of few-shot learning',
      url: 'https://arxiv.org/abs/1904.05046'
    },
    {
      title: 'MAML Tutorial',
      description: 'Guide to Model-Agnostic Meta-Learning',
      url: 'https://arxiv.org/abs/1703.03400'
    }
  ],
  prerequisites: ['Deep Learning', 'Meta-Learning', 'Transfer Learning'],
  relatedTopics: ['Meta-Learning', 'Transfer Learning', 'Zero-Shot Learning']
};