import { NodeContent } from '../../../../types/content';

export const curriculumLearningContent: NodeContent = {
  title: 'Curriculum Learning',
  description: 'A training strategy that structures the learning process from easier to harder examples, mimicking human learning patterns.',
  concepts: [
    'Difficulty Scoring',
    'Sample Ordering',
    'Progressive Training',
    'Task Decomposition',
    'Dynamic Curriculum'
  ],
  examples: [
    {
      language: 'python',
      description: 'Curriculum learning implementation',
      code: `import numpy as np
from torch.utils.data import Sampler

class CurriculumSampler(Sampler):
    def __init__(self, dataset, difficulty_scores, num_epochs):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        self.num_epochs = num_epochs
        self.current_epoch = 0
    
    def get_threshold(self):
        # Progressive threshold based on epoch
        progress = self.current_epoch / self.num_epochs
        return np.quantile(
            self.difficulty_scores,
            min(1.0, progress * 1.5)
        )
    
    def __iter__(self):
        threshold = self.get_threshold()
        indices = np.where(
            self.difficulty_scores <= threshold
        )[0]
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.dataset)`
    }
  ],
  resources: [
    {
      title: 'Curriculum Learning Paper',
      description: 'Original curriculum learning paper',
      url: 'https://dl.acm.org/doi/10.1145/1553374.1553380'
    },
    {
      title: 'Dynamic Curriculum Learning',
      description: 'Advanced curriculum strategies',
      url: 'https://arxiv.org/abs/1910.13719'
    }
  ],
  prerequisites: ['Machine Learning', 'Training Strategies', 'Learning Theory'],
  relatedTopics: ['Transfer Learning', 'Meta-Learning', 'Training Optimization']
};