import { NodeContent } from '../../../../types/content';

export const metricsContent: NodeContent = {
  title: 'Evaluation Metrics',
  description: 'Measures used to assess the performance of machine learning models across different tasks and domains.',
  concepts: [
    'Accuracy and Precision',
    'Recall and F1-Score',
    'ROC and AUC',
    'Mean Average Precision',
    'Custom Metrics'
  ],
  examples: [
    {
      language: 'python',
      description: 'Implementing common metrics',
      code: `from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average='weighted'
)

# ROC AUC for binary classification
auc = roc_auc_score(y_true, y_pred_proba)

# Custom metric
def custom_metric(y_true, y_pred):
    return some_calculation(y_true, y_pred)`
    }
  ],
  resources: [
    {
      title: 'Metrics Guide',
      description: 'Comprehensive guide to ML metrics',
      url: 'https://scikit-learn.org/stable/modules/model_evaluation.html'
    },
    {
      title: 'Beyond Accuracy',
      description: 'Advanced evaluation metrics',
      url: 'https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/'
    }
  ],
  prerequisites: ['Statistics', 'Machine Learning Basics', 'Model Evaluation'],
  relatedTopics: ['Model Evaluation', 'Performance Analysis', 'Model Selection']
};