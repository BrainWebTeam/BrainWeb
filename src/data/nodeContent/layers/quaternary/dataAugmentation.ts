import { NodeContent } from '../../../../types/content';

export const dataAugmentationContent: NodeContent = {
  title: 'Data Augmentation',
  description: 'Techniques to artificially increase the size and diversity of training data by applying various transformations while preserving the semantic content.',
  concepts: [
    'Image Transformations',
    'Text Augmentation',
    'Audio Augmentation',
    'Mixup and CutMix',
    'Random Erasing'
  ],
  examples: [
    {
      language: 'python',
      description: 'Image augmentation with torchvision',
      code: `from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])`
    }
  ],
  resources: [
    {
      title: 'Data Augmentation Review',
      description: 'Comprehensive survey of augmentation techniques',
      url: 'https://arxiv.org/abs/1904.12848'
    },
    {
      title: 'Advanced Augmentation',
      description: 'Modern data augmentation methods',
      url: 'https://pytorch.org/vision/stable/transforms.html'
    }
  ],
  prerequisites: ['Computer Vision', 'Deep Learning', 'Image Processing'],
  relatedTopics: ['Image Processing', 'Training Techniques', 'Regularization']
};