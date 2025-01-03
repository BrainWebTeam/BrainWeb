import { NodeContent } from '../../../../types/content';

export const optimizationContent: NodeContent = {
  title: 'Optimization',
  description: 'Mathematical and computational techniques for finding the best solution from a set of alternatives, crucial for training machine learning models.',
  concepts: [
    'Gradient Descent',
    'Convex Optimization',
    'Hyperparameter Tuning',
    'Constraint Optimization',
    'Global vs Local Optima'
  ],
  examples: [
    {
      language: 'python',
      description: 'Gradient descent implementation',
      code: `import numpy as np

def gradient_descent(f, df, x0, learning_rate=0.01, n_iter=100):
    x = x0
    history = [x]
    
    for _ in range(n_iter):
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append(x)
    
    return x, history

# Example usage for minimizing x^2
f = lambda x: x**2
df = lambda x: 2*x

optimal_x, history = gradient_descent(f, df, x0=2.0)`
    }
  ],
  resources: [
    {
      title: 'Convex Optimization',
      description: 'Stanford\'s course on convex optimization',
      url: 'https://web.stanford.edu/~boyd/cvxbook/'
    },
    {
      title: 'Optimization Methods',
      description: 'Deep learning optimization techniques',
      url: 'https://d2l.ai/chapter_optimization/'
    }
  ],
  prerequisites: ['Calculus', 'Linear Algebra', 'Programming'],
  relatedTopics: ['Neural Networks', 'Model Training', 'Loss Functions']
};