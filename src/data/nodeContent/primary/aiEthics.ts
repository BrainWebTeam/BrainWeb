import { NodeContent } from '../../../types/content';

export const aiEthicsContent: NodeContent = {
  title: 'AI Ethics',
  description: 'The study and implementation of moral principles and guidelines in the development and deployment of artificial intelligence systems.',
  concepts: [
    'Fairness and Bias',
    'Transparency and Explainability',
    'Privacy and Data Protection',
    'Accountability',
    'Social Impact'
  ],
  examples: [
    {
      language: 'python',
      description: 'Bias detection in ML models',
      code: `from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

# Check for bias in dataset
metrics = BinaryLabelDatasetMetric(
    dataset, 
    unprivileged_groups=[{'sex': 0}],
    privileged_groups=[{'sex': 1}]
)

# Calculate disparate impact
di = metrics.disparate_impact()`
    }
  ],
  resources: [
    {
      title: 'AI Ethics Guidelines',
      description: 'IEEE Ethics Guidelines for AI',
      url: 'https://standards.ieee.org/industry-connections/ec/autonomous-systems.html'
    },
    {
      title: 'AI Ethics Course',
      description: 'Fast.ai course on AI Ethics',
      url: 'https://ethics.fast.ai/'
    }
  ],
  prerequisites: ['AI Fundamentals', 'Social Sciences', 'Philosophy of Technology'],
  relatedTopics: ['Responsible AI', 'AI Governance', 'AI Safety']
};