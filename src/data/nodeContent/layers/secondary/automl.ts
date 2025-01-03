import { NodeContent } from '../../../../types/content';

export const automlContent: NodeContent = {
  title: 'AutoML',
  description: 'Automated Machine Learning systems that handle the end-to-end process of applying machine learning to real-world problems with minimal human intervention.',
  concepts: [
    'Hyperparameter Optimization',
    'Neural Architecture Search',
    'Feature Selection',
    'Model Selection',
    'Pipeline Optimization'
  ],
  examples: [
    {
      language: 'python',
      description: 'AutoML with auto-sklearn',
      code: `from autosklearn.classification import AutoSklearnClassifier

# Create and train AutoML model
automl = AutoSklearnClassifier(
    time_left_for_this_task=300,
    per_run_time_limit=30,
    ensemble_size=1
)

# Fit model
automl.fit(X_train, y_train)

# Get best model
print(automl.leaderboard())
print("Best model:", automl.show_models())`
    }
  ],
  resources: [
    {
      title: 'Auto-sklearn Documentation',
      description: 'Automated machine learning toolkit',
      url: 'https://automl.github.io/auto-sklearn/'
    },
    {
      title: 'Google Cloud AutoML',
      description: 'Enterprise AutoML platform',
      url: 'https://cloud.google.com/automl'
    }
  ],
  prerequisites: ['Machine Learning', 'Python Programming', 'Data Science'],
  relatedTopics: ['Model Selection', 'Hyperparameter Tuning', 'Neural Architecture Search']
};