import { NodeContent } from '../../../../types/content';

export const gradientDescentContent: NodeContent = {
  title: 'Gradient Descent',
  description: 'An optimization algorithm used to minimize the loss function in machine learning models by iteratively moving toward the minimum.',
  concepts: [
    'Batch Gradient Descent',
    'Stochastic Gradient Descent',
    'Mini-batch Gradient Descent',
    'Learning Rate',
    'Momentum'
  ],
  examples: [
    {
      language: 'python',
      description: 'Simple gradient descent implementation',
      code: `import numpy as np

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])
    
    for _ in range(n_iterations):
        # Compute predictions
        h = np.dot(X, theta)
        
        # Compute gradients
        gradients = (1/m) * np.dot(X.T, (h - y))
        
        # Update parameters
        theta -= learning_rate * gradients
    
    return theta`
    }
  ],
  resources: [
    {
      title: 'Gradient Descent Guide',
      description: 'Visual guide to gradient descent',
      url: 'https://ruder.io/optimizing-gradient-descent/'
    },
    {
      title: 'Optimization Algorithms',
      description: 'Deep dive into optimization methods',
      url: 'https://d2l.ai/chapter_optimization/'
    }
  ],
  prerequisites: ['Calculus', 'Linear Algebra', 'Machine Learning Basics'],
  relatedTopics: ['Optimization', 'Loss Functions', 'Neural Networks']
};