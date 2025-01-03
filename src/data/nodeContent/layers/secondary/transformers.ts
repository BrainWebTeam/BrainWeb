import { NodeContent } from '../../../../types/content';

export const transformersContent: NodeContent = {
  title: 'Transformers',
  description: 'A neural network architecture that uses self-attention mechanisms to process sequential data, revolutionizing natural language processing tasks.',
  concepts: [
    'Self-Attention Mechanism',
    'Multi-Head Attention',
    'Positional Encoding',
    'Encoder-Decoder Architecture',
    'Transfer Learning'
  ],
  examples: [
    {
      language: 'python',
      description: 'Using transformers with Hugging Face',
      code: `from transformers import AutoTokenizer, AutoModel

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Process text
text = "Understanding transformers is fascinating!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)`
    }
  ],
  resources: [
    {
      title: 'Attention Is All You Need',
      description: 'Original transformer paper',
      url: 'https://arxiv.org/abs/1706.03762'
    },
    {
      title: 'Hugging Face Documentation',
      description: 'Comprehensive guide to using transformers',
      url: 'https://huggingface.co/docs'
    }
  ],
  prerequisites: ['Deep Learning', 'NLP Basics', 'Python Programming'],
  relatedTopics: ['BERT', 'GPT', 'Attention Mechanism']
};