import { NodeContent } from '../../../../types/content';

export const zeroShotLearningContent: NodeContent = {
  title: 'Zero-Shot Learning',
  description: 'The ability of a model to recognize or classify objects it has never seen during training by leveraging semantic descriptions or relationships.',
  concepts: [
    'Semantic Embeddings',
    'Cross-Modal Transfer',
    'Attribute Learning',
    'Semantic Space Mapping',
    'Visual-Semantic Embeddings'
  ],
  examples: [
    {
      language: 'python',
      description: 'Zero-shot classification with CLIP',
      code: `from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare image and candidate labels
inputs = processor(
    images=image,
    text=["a dog", "a cat", "a bird"],
    return_tensors="pt",
    padding=True
)

# Get prediction
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)`
    }
  ],
  resources: [
    {
      title: 'Zero-Shot Learning Review',
      description: 'Comprehensive survey of zero-shot learning',
      url: 'https://arxiv.org/abs/1707.00600'
    },
    {
      title: 'CLIP Paper',
      description: 'OpenAI CLIP model for zero-shot learning',
      url: 'https://arxiv.org/abs/2103.00020'
    }
  ],
  prerequisites: ['Deep Learning', 'Transfer Learning', 'Embeddings'],
  relatedTopics: ['Few-Shot Learning', 'Transfer Learning', 'Semantic Embeddings']
};