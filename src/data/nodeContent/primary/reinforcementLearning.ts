import { NodeContent } from '../../../types/content';

export const reinforcementLearningContent: NodeContent = {
  title: 'Reinforcement Learning',
  description: 'A type of machine learning where agents learn to make decisions by interacting with an environment and receiving rewards or penalties.',
  concepts: [
    'Markov Decision Processes',
    'Q-Learning',
    'Policy Gradient Methods',
    'Value Functions',
    'Exploration vs Exploitation'
  ],
  examples: [
    {
      language: 'python',
      description: 'Simple Q-Learning implementation',
      code: `import numpy as np

class QLearning:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9):
        self.q_table = np.zeros((states, actions))
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
    
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-learning formula
        new_value = (1 - self.alpha) * old_value + \
                   self.alpha * (reward + self.gamma * next_max)
        
        self.q_table[state, action] = new_value`
    }
  ],
  resources: [
    {
      title: 'Spinning Up in Deep RL',
      description: 'OpenAI\'s educational resource for deep RL',
      url: 'https://spinningup.openai.com/'
    },
    {
      title: 'RL Course by David Silver',
      description: 'Comprehensive course on RL fundamentals',
      url: 'https://www.davidsilver.uk/teaching/'
    }
  ],
  prerequisites: ['Probability Theory', 'Python Programming', 'Basic ML Concepts'],
  relatedTopics: ['Deep RL', 'Game Theory', 'Multi-agent Systems']
};