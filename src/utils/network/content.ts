interface Resource {
  title: string;
  description: string;
  url: string;
}

interface NodeContent {
  title: string;
  description: string;
  concepts: string[];
  examples: string[];
  resources: Resource[];
}

type NodeContentMap = {
  [key: string]: {
    [key: string]: NodeContent;
  };
};

export const NODE_CONTENT: NodeContentMap = {
  center: {
    'center': {
      title: 'Artificial Intelligence',
      description: 'Artificial Intelligence (AI) is the simulation of human intelligence by machines. It encompasses various subfields that enable computers to perform tasks that typically require human intelligence.',
      concepts: [
        'Machine Learning and Pattern Recognition',
        'Natural Language Processing',
        'Computer Vision',
        'Robotics and Automation',
        'Expert Systems and Knowledge Representation'
      ],
      examples: [
        'Virtual assistants like Siri and Alexa',
        'Self-driving cars and autonomous systems',
        'AI-powered recommendation systems',
        'Medical diagnosis and healthcare AI',
        'Game-playing AI like AlphaGo'
      ],
      resources: [
        {
          title: 'AI: A Modern Approach',
          description: 'The standard text in AI, used in universities worldwide',
          url: 'https://aima.cs.berkeley.edu/'
        },
        {
          title: 'MIT OpenCourseWare: Artificial Intelligence',
          description: 'Free AI course materials from MIT',
          url: 'https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/'
        }
      ]
    }
  },
  primary: {
    'primary-0': {
      title: 'Machine Learning',
      description: 'Machine Learning is a subset of AI that focuses on developing systems that can learn from and make decisions based on data.',
      concepts: [
        'Supervised Learning',
        'Unsupervised Learning',
        'Reinforcement Learning',
        'Model Training and Evaluation',
        'Feature Engineering'
      ],
      examples: [
        'Email spam detection',
        'Product recommendations',
        'Credit card fraud detection',
        'Image classification',
        'Weather prediction'
      ],
      resources: [
        {
          title: 'Machine Learning Crash Course',
          description: 'Google\'s fast-paced, practical introduction to ML',
          url: 'https://developers.google.com/machine-learning/crash-course'
        }
      ]
    }
    // Add more primary nodes content...
  }
  // Add content for other node types...
};