import { NodeContent } from '../../../../types/content';

export const backpropagationContent: NodeContent = {
  title: 'Backpropagation',
  description: 'The primary algorithm for training neural networks by calculating gradients of the loss function with respect to the network weights.',
  concepts: [
    'Chain Rule',
    'Forward Pass',
    'Backward Pass',
    'Weight Updates',
    'Gradient Flow'
  ],
  examples: [
    {
      language: 'python',
      description: 'Neural network with manual backpropagation',
      code: `import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.weights = [np.random.randn(y, x) 
                       for x, y in zip(layers[:-1], layers[1:])]
    
    def forward(self, x):
        activations = [x]
        for w in self.weights:
            x = self.sigmoid(np.dot(w, x))
            activations.append(x)
        return activations
    
    def backward(self, x, y, activations):
        deltas = []
        error = activations[-1] - y
        delta = error * self.sigmoid_prime(activations[-1])
        deltas.append(delta)
        
        for l in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(self.weights[l].T, delta) * \
                   self.sigmoid_prime(activations[l])
            deltas.append(delta)
        
        return deltas[::-1]`
    }
  ],
  resources: [
    {
      title: 'Backpropagation Explained',
      description: 'Visual guide to backpropagation',
      url: 'https://colah.github.io/posts/2015-08-Backprop/'
    },
    {
      title: '3Blue1Brown Neural Networks',
      description: 'Visual explanation of neural networks',
      url: 'https://www.3blue1brown.com/lessons/neural-networks'
    }
  ],
  prerequisites: ['Calculus', 'Linear Algebra', 'Neural Networks'],
  relatedTopics: ['Gradient Descent', 'Neural Networks', 'Optimization']
};