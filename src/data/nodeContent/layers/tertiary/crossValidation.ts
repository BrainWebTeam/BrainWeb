import { NodeContent } from '../../../../types/content';

export const crossValidationContent: NodeContent = {
  title: 'Cross-Validation',
  description: 'A resampling method used to assess model performance and ensure reliable evaluation on unseen data.',
  concepts: [
    'K-Fold Cross-Validation',
    'Stratified K-Fold',
    'Leave-One-Out',
    'Time Series Split',
    'Nested Cross-Validation'
  ],
  examples: [
    {
      language: 'python',
      description: 'K-fold cross-validation implementation',
      code: `from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Create cross-validation object
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring='accuracy'
)

print(f"CV Scores: {scores}")
print(f"Mean CV Score: {scores.mean():.3f}")`
    }
  ],
  resources: [
    {
      title: 'Cross-Validation Guide',
      description: 'Comprehensive guide to cross-validation',
      url: 'https://scikit-learn.org/stable/modules/cross_validation.html'
    },
    {
      title: 'Time Series Cross-Validation',
      description: 'Special considerations for time series data',
      url: 'https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4'
    }
  ],
  prerequisites: ['Statistics', 'Machine Learning Basics', 'Model Evaluation'],
  relatedTopics: ['Model Validation', 'Performance Metrics', 'Hyperparameter Tuning']
};