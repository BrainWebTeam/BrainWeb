import { NodeContent } from '../../../../types/content';

export const naturalLanguageContent: NodeContent = {
  title: 'Natural Language Processing',
  description: 'The branch of AI focused on enabling computers to understand, interpret, and generate human language in useful ways.',
  concepts: [
    'Text Processing',
    'Language Models',
    'Named Entity Recognition',
    'Sentiment Analysis',
    'Machine Translation'
  ],
  examples: [
    {
      language: 'python',
      description: 'Basic NLP with spaCy',
      code: `import spacy

# Load English language model
nlp = spacy.load('en_core_web_sm')

# Process text
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Named Entity Recognition
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# Part-of-speech tagging
for token in doc:
    print(f"{token.text}: {token.pos_}")`
    }
  ],
  resources: [
    {
      title: 'NLP Course',
      description: 'Stanford CS224N NLP Course',
      url: 'http://web.stanford.edu/class/cs224n/'
    },
    {
      title: 'Hugging Face',
      description: 'State-of-the-art NLP library',
      url: 'https://huggingface.co/'
    }
  ],
  prerequisites: ['Linguistics', 'Machine Learning', 'Programming'],
  relatedTopics: ['Transformers', 'BERT', 'GPT', 'Text Mining']
};