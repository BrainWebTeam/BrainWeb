import { NodeContent } from '../../../../types/content';

export const abTestingContent: NodeContent = {
  title: 'A/B Testing',
  description: 'A statistical method for comparing two versions of a model or system to determine which performs better in production.',
  concepts: [
    'Statistical Significance',
    'Traffic Splitting',
    'Hypothesis Testing',
    'Metrics Definition',
    'Experiment Design'
  ],
  examples: [
    {
      language: 'python',
      description: 'A/B testing implementation',
      code: `from scipy import stats
import numpy as np

class ABTest:
    def __init__(self, control_group, test_group):
        self.control = control_group
        self.test = test_group
    
    def calculate_significance(self):
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(
            self.control,
            self.test
        )
        
        # Calculate effect size
        effect_size = (np.mean(self.test) - np.mean(self.control)) / \
                     np.std(self.control)
        
        return {
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05
        }`
    }
  ],
  resources: [
    {
      title: 'A/B Testing Guide',
      description: 'Comprehensive guide to A/B testing',
      url: 'https://www.optimizely.com/optimization-glossary/ab-testing/'
    },
    {
      title: 'Statistical Testing',
      description: 'Statistical methods for A/B testing',
      url: 'https://www.evanmiller.org/ab-testing/'
    }
  ],
  prerequisites: ['Statistics', 'Experimental Design', 'Data Analysis'],
  relatedTopics: ['Hypothesis Testing', 'Model Evaluation', 'Experimentation']
};