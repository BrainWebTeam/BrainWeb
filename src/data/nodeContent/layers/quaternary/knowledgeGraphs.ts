import { NodeContent } from '../../../../types/content';

export const knowledgeGraphsContent: NodeContent = {
  title: 'Knowledge Graphs',
  description: 'Structured representations of knowledge that capture entities and their relationships, enabling reasoning and inference in AI systems.',
  concepts: [
    'Entity-Relationship Modeling',
    'Graph Embeddings',
    'Semantic Relations',
    'Reasoning Engines',
    'Knowledge Integration'
  ],
  examples: [
    {
      language: 'python',
      description: 'Knowledge graph construction and querying',
      code: `from rdflib import Graph, Literal, RDF, URIRef

# Create a knowledge graph
g = Graph()

# Define entities and relationships
person = URIRef("http://example.org/person")
knows = URIRef("http://example.org/knows")
name = URIRef("http://example.org/name")

# Add triples
g.add((person, RDF.type, URIRef("http://example.org/Person")))
g.add((person, name, Literal("John Doe")))
g.add((person, knows, URIRef("http://example.org/jane")))

# Query the graph
for s, p, o in g:
    print(f"{s} {p} {o}")`
    }
  ],
  resources: [
    {
      title: 'Knowledge Graphs',
      description: 'Comprehensive guide to knowledge graphs',
      url: 'https://arxiv.org/abs/2003.02320'
    },
    {
      title: 'Graph Neural Networks',
      description: 'Deep learning on knowledge graphs',
      url: 'https://distill.pub/2021/gnn-intro/'
    }
  ],
  prerequisites: ['Graph Theory', 'Semantic Web', 'Machine Learning'],
  relatedTopics: ['Graph Neural Networks', 'Semantic Web', 'Ontologies']
};