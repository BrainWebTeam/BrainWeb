import { NodeContent } from '../../../../types/content';

export const activeLearningContent: NodeContent = {
  title: 'Active Learning',
  description: 'A machine learning approach where the algorithm actively selects the most informative samples for labeling, reducing the amount of labeled data needed for training.',
  concepts: [
    'Uncertainty Sampling',
    'Query Strategies',
    'Pool-based Sampling',
    'Stream-based Sampling',
    'Query by Committee'
  ],
  examples: [
    {
      language: 'python',
      description: 'Uncertainty sampling implementation',
      code: `from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier

# Initialize active learner
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=uncertainty_sampling,
    X_training=X_initial,
    y_training=y_initial
)

# Query for most uncertain instance
query_idx, query_inst = learner.query(X_pool)

# Update model with new labeled data
learner.teach(
    X=X_pool[query_idx],
    y=y_pool[query_idx]
)`
    }
  ],
  resources: [
    {
      title: 'Active Learning Literature Survey',
      description: 'Comprehensive guide to active learning',
      url: 'https://minds.wisconsin.edu/handle/1793/60660'
    },
    {
      title: 'ModAL Documentation',
      description: 'Active learning framework for Python',
      url: 'https://modal-python.readthedocs.io/'
    }
  ],
  prerequisites: ['Machine Learning', 'Supervised Learning', 'Model Evaluation'],
  relatedTopics: ['Semi-Supervised Learning', 'Model Selection', 'Data Efficiency']
};