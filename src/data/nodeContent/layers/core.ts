import { NodeContent } from '../../../types/content';

export const aiCoreContent: NodeContent = {
  title: 'AI Core',
  description: 'The foundational concepts and principles of Artificial Intelligence, encompassing the broad spectrum of technologies and methodologies that enable machines to simulate human intelligence.',
  concepts: [
    'Foundations of Artificial Intelligence',
    'Types of AI Systems',
    'Problem Solving and Search',
    'Knowledge Representation',
    'Planning and Decision Making'
  ],
  examples: [
    {
      language: 'python',
      description: 'Basic AI agent structure',
      code: `class AIAgent:
    def __init__(self):
        self.knowledge_base = {}
        self.goals = []
    
    def perceive(self, environment):
        # Update knowledge based on environment
        pass
    
    def think(self):
        # Process information and make decisions
        pass
    
    def act(self):
        # Take action based on decisions
        pass`
    }
  ],
  resources: [
    {
      title: 'AI: A Modern Approach',
      description: 'The definitive textbook on artificial intelligence',
      url: 'https://aima.cs.berkeley.edu/'
    },
    {
      title: 'MIT OpenCourseWare: AI',
      description: 'Comprehensive AI course materials',
      url: 'https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/'
    }
  ],
  prerequisites: [],
  relatedTopics: ['Machine Learning', 'Neural Networks', 'Expert Systems']
};