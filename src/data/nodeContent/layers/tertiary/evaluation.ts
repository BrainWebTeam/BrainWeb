import { NodeContent } from '../../../../types/content';

export const evaluationContent: NodeContent = {
  title: 'Model Evaluation',
  description: 'The process of assessing machine learning model performance, reliability, and generalization ability using various techniques and metrics.',
  concepts: [
    'Cross-Validation',
    'Hold-out Validation',
    'Error Analysis',
    'Model Comparison',
    'Performance Metrics'
  ],
  examples: [
    {
      language: 'python',
      description: 'Model evaluation techniques',
      code: `from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    learning_curve
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cross-validation
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)

# Learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5
)`
    }
  ],
  resources: [
    {
      title: 'Model Evaluation Guide',
      description: 'Comprehensive guide to model evaluation',
      url: 'https://scikit-learn.org/stable/modules/model_evaluation.html'
    },
    {
      title: 'Beyond Accuracy',
      description: 'Advanced evaluation techniques',
      url: 'https://machinelearningmastery.com/how-to-evaluate-machine-learning-algorithms/'
    }
  ],
  prerequisites: ['Machine Learning', 'Statistics', 'Data Analysis'],
  relatedTopics: ['Cross-Validation', 'Metrics', 'Model Selection']
};