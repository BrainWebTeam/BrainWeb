import { NodeContent } from '../../../../types/content';

export const machineLearningContent: NodeContent = {
  title: 'Machine Learning',
  description: 'A subset of AI that enables systems to learn and improve from experience without being explicitly programmed.',
  concepts: [
    'Supervised Learning',
    'Unsupervised Learning',
    'Reinforcement Learning',
    'Model Training and Evaluation',
    'Feature Engineering'
  ],
  examples: [
    {
      language: 'python',
      description: 'Simple linear regression',
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
    }
  ],
  resources: [
    {
      title: 'Machine Learning Crash Course',
      description: 'Google\'s fast-paced, practical introduction to ML',
      url: 'https://developers.google.com/machine-learning/crash-course'
    },
    {
      title: 'Fast.ai',
      description: 'Practical Deep Learning for Coders',
      url: 'https://www.fast.ai/'
    }
  ],
  prerequisites: ['Python Programming', 'Statistics', 'Linear Algebra'],
  relatedTopics: ['Deep Learning', 'Neural Networks', 'Data Science']
};