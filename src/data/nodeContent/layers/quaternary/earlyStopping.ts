import { NodeContent } from '../../../../types/content';

export const earlyStoppingContent: NodeContent = {
  title: 'Early Stopping',
  description: 'A regularization technique that stops training when the model performance on a validation set stops improving, preventing overfitting.',
  concepts: [
    'Validation Monitoring',
    'Patience Parameter',
    'Model Selection',
    'Overfitting Prevention',
    'Best Model Tracking'
  ],
  examples: [
    {
      language: 'python',
      description: 'Early stopping implementation',
      code: `class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0`
    }
  ],
  resources: [
    {
      title: 'Early Stopping Guide',
      description: 'Comprehensive guide to early stopping',
      url: 'https://scikit-learn.org/stable/modules/early_stopping.html'
    },
    {
      title: 'Model Selection',
      description: 'Best practices for model selection',
      url: 'https://pytorch.org/tutorials/beginner/saving_loading_models.html'
    }
  ],
  prerequisites: ['Model Training', 'Validation Techniques', 'Overfitting'],
  relatedTopics: ['Model Selection', 'Regularization', 'Cross-Validation']
};