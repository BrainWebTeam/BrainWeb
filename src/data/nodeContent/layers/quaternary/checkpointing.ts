import { NodeContent } from '../../../../types/content';

export const checkpointingContent: NodeContent = {
  title: 'Checkpointing',
  description: 'A technique to save model states during training, allowing recovery from interruptions and selection of the best performing model.',
  concepts: [
    'Model State Saving',
    'Training Recovery',
    'Best Model Selection',
    'Memory Management',
    'Distributed Training'
  ],
  examples: [
    {
      language: 'python',
      description: 'Model checkpointing implementation',
      code: `import torch

def save_checkpoint(model, optimizer, epoch, best_val_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, f'checkpoint_epoch_{epoch}.pt')

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_val_loss']`
    }
  ],
  resources: [
    {
      title: 'PyTorch Checkpointing',
      description: 'Official guide to model checkpointing',
      url: 'https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html'
    },
    {
      title: 'Distributed Checkpointing',
      description: 'Advanced checkpointing techniques',
      url: 'https://pytorch.org/docs/stable/distributed.html'
    }
  ],
  prerequisites: ['Model Training', 'File I/O', 'State Management'],
  relatedTopics: ['Model Serialization', 'Training Recovery', 'Model Selection']
};