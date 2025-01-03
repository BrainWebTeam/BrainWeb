import { NodeContent } from '../../../../types/content';

export const adversarialTrainingContent: NodeContent = {
  title: 'Adversarial Training',
  description: 'A training technique that improves model robustness by incorporating adversarial examples into the training process.',
  concepts: [
    'Adversarial Examples',
    'FGSM Attack',
    'PGD Training',
    'Robust Optimization',
    'Defense Mechanisms'
  ],
  examples: [
    {
      language: 'python',
      description: 'FGSM adversarial training',
      code: `import torch
import torch.nn.functional as F

def fgsm_attack(model, x, y, epsilon):
    x.requires_grad = True
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    loss.backward()
    
    # Create perturbation
    perturbation = epsilon * x.grad.sign()
    x_adv = x + perturbation
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv

# Training loop with adversarial examples
def train_adversarial(model, loader, optimizer, epsilon):
    for x, y in loader:
        # Generate adversarial examples
        x_adv = fgsm_attack(model, x, y, epsilon)
        
        # Train on both clean and adversarial
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y) + \
               F.cross_entropy(model(x_adv), y)
        loss.backward()
        optimizer.step()`
    }
  ],
  resources: [
    {
      title: 'Adversarial Training',
      description: 'Guide to adversarial training methods',
      url: 'https://arxiv.org/abs/1412.6572'
    },
    {
      title: 'Robust Deep Learning',
      description: 'Survey of robustness in deep learning',
      url: 'https://arxiv.org/abs/1906.06032'
    }
  ],
  prerequisites: ['Deep Learning', 'Optimization', 'Model Security'],
  relatedTopics: ['Model Robustness', 'Security', 'Adversarial Attacks']
};