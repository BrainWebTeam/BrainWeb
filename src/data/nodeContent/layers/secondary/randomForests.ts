import { NodeContent } from '../../../../types/content';

export const randomForestsContent: NodeContent = {
  title: 'Random Forests',
  description: 'An ensemble learning method that constructs multiple decision trees and combines their predictions for more robust and accurate results.',
  concepts: [
    'Bagging',
    'Feature Randomness',
    'Ensemble Learning',
    'Out-of-Bag Error',
    'Feature Importance'
  ],
  examples: [
    {
      language: 'python',
      description: 'Random forest implementation',
      code: `from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create and train model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5
)
rf.fit(X_train, y_train)

# Feature importance
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)
for idx in sorted_idx:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}")`
    }
  ],
  resources: [
    {
      title: 'Random Forests Guide',
      description: 'In-depth guide to random forests',
      url: 'https://scikit-learn.org/stable/modules/ensemble.html#random-forests'
    },
    {
      title: 'Random Forest Visualization',
      description: 'Interactive visualization of random forests',
      url: 'https://explained.ai/decision-tree-viz/index.html'
    }
  ],
  prerequisites: ['Decision Trees', 'Statistics', 'Ensemble Methods'],
  relatedTopics: ['Gradient Boosting', 'Feature Selection', 'Model Interpretation']
};