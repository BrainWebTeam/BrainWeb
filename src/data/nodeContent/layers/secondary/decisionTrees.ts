import { NodeContent } from '../../../../types/content';

export const decisionTreesContent: NodeContent = {
  title: 'Decision Trees',
  description: 'A tree-like model that makes decisions based on asking a series of questions about the features, creating a flowchart-like structure for classification or regression.',
  concepts: [
    'Information Gain',
    'Entropy and Gini Impurity',
    'Tree Pruning',
    'Feature Selection',
    'Split Criteria'
  ],
  examples: [
    {
      language: 'python',
      description: 'Decision tree classifier with scikit-learn',
      code: `from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Create and train model
tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    criterion='gini'
)
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)

# Visualize tree structure
from sklearn.tree import plot_tree
plot_tree(tree, feature_names=feature_names)`
    }
  ],
  resources: [
    {
      title: 'Decision Trees Guide',
      description: 'Comprehensive guide to decision trees',
      url: 'https://scikit-learn.org/stable/modules/tree.html'
    },
    {
      title: 'Interactive Decision Tree',
      description: 'Visual learning tool for decision trees',
      url: 'http://www.r2d3.us/visual-intro-to-machine-learning-part-1/'
    }
  ],
  prerequisites: ['Statistics', 'Probability Theory', 'Basic ML Concepts'],
  relatedTopics: ['Random Forests', 'Ensemble Methods', 'Feature Selection']
};