import { NodeContent } from '../../../../types/content';

export const hyperparameterTuningContent: NodeContent = {
  title: 'Hyperparameter Tuning',
  description: 'The process of optimizing model hyperparameters to improve performance and generalization ability.',
  concepts: [
    'Grid Search',
    'Random Search',
    'Bayesian Optimization',
    'Cross-Validation',
    'Parameter Spaces'
  ],
  examples: [
    {
      language: 'python',
      description: 'Grid search with scikit-learn',
      code: `from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.1, 1],
}

# Create grid search
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

# Fit and get best parameters
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)`
    }
  ],
  resources: [
    {
      title: 'Hyperparameter Tuning Guide',
      description: 'Comprehensive guide to tuning ML models',
      url: 'https://scikit-learn.org/stable/modules/grid_search.html'
    },
    {
      title: 'Optuna Documentation',
      description: 'Advanced hyperparameter optimization framework',
      url: 'https://optuna.org/'
    }
  ],
  prerequisites: ['Machine Learning Basics', 'Cross-Validation', 'Model Evaluation'],
  relatedTopics: ['Model Selection', 'AutoML', 'Performance Optimization']
};