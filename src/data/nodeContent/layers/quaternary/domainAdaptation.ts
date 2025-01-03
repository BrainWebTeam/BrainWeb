import { NodeContent } from '../../../../types/content';

export const domainAdaptationContent: NodeContent = {
  title: 'Domain Adaptation',
  description: 'Techniques for adapting models trained on one domain to perform well on a different but related target domain.',
  concepts: [
    'Domain Shift',
    'Feature Alignment',
    'Adversarial Adaptation',
    'Domain Invariance',
    'Transfer Learning'
  ],
  examples: [
    {
      language: 'python',
      description: 'Domain adversarial training',
      code: `import torch.nn as nn

class DomainAdversarial(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.task_classifier = nn.Linear(128, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Source vs Target
            nn.GradientReversal()  # Custom layer for gradient reversal
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        task_out = self.task_classifier(features)
        domain_out = self.domain_classifier(features)
        return task_out, domain_out`
    }
  ],
  resources: [
    {
      title: 'Domain Adaptation Survey',
      description: 'Comprehensive overview of domain adaptation',
      url: 'https://arxiv.org/abs/1812.11806'
    },
    {
      title: 'DANN Paper',
      description: 'Domain adversarial neural networks',
      url: 'https://arxiv.org/abs/1505.07818'
    }
  ],
  prerequisites: ['Transfer Learning', 'Deep Learning', 'Adversarial Training'],
  relatedTopics: ['Transfer Learning', 'Domain Shift', 'Adversarial Training']
};