import { NodeContent } from '../../../../types/content';

export const knowledgeDistillationContent: NodeContent = {
  title: 'Knowledge Distillation',
  description: 'A model compression technique where a smaller student model learns to mimic the behavior of a larger teacher model.',
  concepts: [
    'Teacher-Student Architecture',
    'Soft Targets',
    'Temperature Scaling',
    'Dark Knowledge',
    'Ensemble Distillation'
  ],
  examples: [
    {
      language: 'python',
      description: 'Knowledge distillation implementation',
      code: `import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    
    # Distillation loss
    distill_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        soft_targets,
        reduction='batchmean'
    ) * (T * T)
    
    # Student loss
    student_loss = F.cross_entropy(student_logits, labels)
    
    # Combined loss
    return alpha * student_loss + (1 - alpha) * distill_loss`
    }
  ],
  resources: [
    {
      title: 'Distilling Knowledge in Neural Networks',
      description: 'Original paper by Hinton et al.',
      url: 'https://arxiv.org/abs/1503.02531'
    },
    {
      title: 'Knowledge Distillation Methods',
      description: 'Survey of distillation techniques',
      url: 'https://arxiv.org/abs/2006.05525'
    }
  ],
  prerequisites: ['Neural Networks', 'Model Training', 'Loss Functions'],
  relatedTopics: ['Model Compression', 'Transfer Learning', 'Ensemble Learning']
};