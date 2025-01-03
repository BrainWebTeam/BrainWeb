import { NodeContent } from '../../../../types/content';

export const fineTuningContent: NodeContent = {
  title: 'Fine-tuning',
  description: 'The process of taking a pre-trained model and further training it on a specific task or domain to improve its performance for that particular use case.',
  concepts: [
    'Transfer Learning',
    'Layer Freezing',
    'Learning Rate Selection',
    'Catastrophic Forgetting',
    'Domain Adaptation'
  ],
  examples: [
    {
      language: 'python',
      description: 'Fine-tuning a pre-trained model',
      code: `import torch
from transformers import AutoModelForSequenceClassification

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Freeze base layers
for param in model.base_model.parameters():
    param.requires_grad = False

# Train only classification head
optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=2e-5
)`
    }
  ],
  resources: [
    {
      title: 'Fine-tuning Guide',
      description: 'Hugging Face fine-tuning tutorial',
      url: 'https://huggingface.co/docs/transformers/training'
    },
    {
      title: 'Transfer Learning Tips',
      description: 'Best practices for fine-tuning models',
      url: 'https://www.fast.ai/posts/2020-02-13-fastai-A-Layered-API-for-Deep-Learning.html'
    }
  ],
  prerequisites: ['Deep Learning', 'Transfer Learning', 'Model Training'],
  relatedTopics: ['Transfer Learning', 'Model Adaptation', 'Domain Tuning']
};