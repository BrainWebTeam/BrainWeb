import { NodeContent } from '../../../../types/content';

export const embeddingsContent: NodeContent = {
  title: 'Embeddings',
  description: 'Dense vector representations of discrete data that capture semantic relationships and enable machine learning models to process categorical or text data.',
  concepts: [
    'Word Embeddings',
    'Contextual Embeddings',
    'Document Embeddings',
    'Neural Embeddings',
    'Embedding Spaces'
  ],
  examples: [
    {
      language: 'python',
      description: 'Creating word embeddings with Word2Vec',
      code: `from gensim.models import Word2Vec

# Train word embeddings
model = Word2Vec(
    sentences=text_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Get vector for a word
vector = model.wv['word']

# Find similar words
similar = model.wv.most_similar('word')`
    }
  ],
  resources: [
    {
      title: 'Word Embeddings Guide',
      description: 'Stanford NLP guide to word embeddings',
      url: 'https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs.pdf'
    },
    {
      title: 'Embeddings Tutorial',
      description: 'TensorFlow guide to embeddings',
      url: 'https://www.tensorflow.org/text/guide/word_embeddings'
    }
  ],
  prerequisites: ['Neural Networks', 'NLP Basics', 'Linear Algebra'],
  relatedTopics: ['Word2Vec', 'GloVe', 'BERT Embeddings']
};