import { NodeContent } from '../../../../types/content';

export const learningRateSchedulingContent: NodeContent = {
  title: 'Learning Rate Scheduling',
  description: 'Techniques for dynamically adjusting the learning rate during training to improve convergence and model performance.',
  concepts: [
    'Step Decay',
    'Cosine Annealing',
    'Cyclic Learning Rates',
    'Warm-up Strategies',
    'Adaptive Schedules'
  ],
  examples: [
    {
      language: 'python',
      description: 'Learning rate scheduler implementation',
      code: `import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step decay scheduler
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)

# Cosine annealing scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)

# Training loop
for epoch in range(epochs):
    train_one_epoch()
    scheduler.step()`
    }
  ],
  resources: [
    {
      title: 'Learning Rate Scheduling Guide',
      description: 'Comprehensive guide to LR scheduling',
      url: 'https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate'
    },
    {
      title: 'Cyclical Learning Rates',
      description: 'Paper on cyclical learning rates',
      url: 'https://arxiv.org/abs/1506.01186'
    }
  ],
  prerequisites: ['Optimization', 'Model Training', 'Gradient Descent'],
  relatedTopics: ['Optimization', 'Model Training', 'Hyperparameter Tuning']
};