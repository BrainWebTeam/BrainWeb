export const CENTER = {
  x: 400,
  y: 400,
};

export const RADII = {
  PRIMARY: 150,
  SECONDARY: 250,
  TERTIARY: 350,
  QUATERNARY: 450,
} as const;

export const NODE_COUNTS = {
  PRIMARY: 8,
  SECONDARY: 16,
  TERTIARY: 24,
  QUATERNARY: 32,
} as const;

export const CONNECTION_STRENGTHS = {
  CENTER: 1,
  PRIMARY: 0.8,
  SECONDARY: 0.6,
  TERTIARY: 0.4,
  QUATERNARY: 0.3,
} as const;

export const NODE_LABELS = {
  center: ['AI Core'],
  primary: [
    'Machine Learning',
    'Deep Learning',
    'Natural Language',
    'Computer Vision',
    'Reinforcement Learning',
    'Neural Networks',
    'Data Science',
    'AI Ethics'
  ],
  secondary: [
    'Supervised Learning',
    'Unsupervised Learning',
    'Transformers',
    'CNN',
    'RNN',
    'GAN',
    'Decision Trees',
    'Random Forests',
    'SVM',
    'Clustering',
    'Dimensionality Reduction',
    'Time Series',
    'Recommendation Systems',
    'Anomaly Detection',
    'Optimization',
    'AutoML'
  ],
  tertiary: [
    'Transfer Learning',
    'Feature Engineering',
    'Model Deployment',
    'Hyperparameter Tuning',
    'Cross-Validation',
    'Ensemble Methods',
    'Gradient Descent',
    'Backpropagation',
    'Attention Mechanism',
    'Embeddings',
    'Fine-tuning',
    'Model Compression',
    'Knowledge Distillation',
    'Few-shot Learning',
    'Zero-shot Learning',
    'Active Learning',
    'Meta Learning',
    'Neural Architecture',
    'Regularization',
    'Optimization',
    'Loss Functions',
    'Metrics',
    'Evaluation'
  ],
  quaternary: [
    'Data Augmentation',
    'Batch Normalization',
    'Dropout',
    'Layer Normalization',
    'Weight Initialization',
    'Learning Rate Scheduling',
    'Early Stopping',
    'Checkpointing',
    'Model Serialization',
    'Quantization',
    'Pruning',
    'Knowledge Graphs',
    'Adversarial Training',
    'Curriculum Learning',
    'Multi-task Learning',
    'Domain Adaptation',
    'Continual Learning',
    'Online Learning',
    'Distributed Training',
    'Model Parallelism',
    'Data Parallelism',
    'Pipeline Parallelism',
    'Gradient Accumulation',
    'Mixed Precision',
    'Model Monitoring',
    'A/B Testing',
    'Shadow Deployment',
    'Canary Release',
    'Blue-Green Deployment',
    'Feature Stores',
    'Model Registry',
    'Model Versioning'
  ]
} as const;