import { NodeContent } from '../../types/content';

export const machineLearningContent: NodeContent = {
  title: 'Machine Learning',
  description: 'Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.',
  concepts: [
    'Supervised Learning: Training with labeled data',
    'Unsupervised Learning: Finding patterns in unlabeled data',
    'Reinforcement Learning: Learning through action and reward',
    'Feature Engineering: Selecting and transforming input variables',
    'Model Evaluation: Assessing performance and validation'
  ],
  examples: [
    {
      language: 'python',
      description: 'Simple linear regression example',
      code: `from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict([[5]])`
    },
    {
      language: 'python',
      description: 'Basic classification with scikit-learn',
      code: `from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)`
    }
  ],
  resources: [
    {
      title: 'Machine Learning Crash Course',
      description: 'Google\'s fast-paced, practical introduction to machine learning',
      url: 'https://developers.google.com/machine-learning/crash-course'
    },
    {
      title: 'Fast.ai Practical Deep Learning',
      description: 'Free course that teaches deep learning through practical applications',
      url: 'https://course.fast.ai/'
    },
    {
      title: 'Scikit-learn Documentation',
      description: 'Official documentation for the popular ML library',
      url: 'https://scikit-learn.org/stable/'
    }
  ],
  prerequisites: ['Python Programming', 'Statistics Basics', 'Linear Algebra'],
  relatedTopics: ['Deep Learning', 'Neural Networks', 'Data Science']
};