import React from 'react';
import CodeBlock from '../../CodeBlock';

function EmbeddingsTools() {
  return (
    <section id="embeddings-tools">
      <h2>Embeddings Tools</h2>
      <p>
        Tools for generating and managing vector embeddings for text, images, and other data types.
      </p>

      <CodeBlock
        language="typescript"
        code={`interface EmbeddingsConfig {
  model: {
    type: 'transformer' | 'siamese' | 'custom';
    dimensions: number;
    pooling: 'mean' | 'max' | 'cls';
  };
  preprocessing: {
    text?: {
      tokenization: 'word' | 'subword' | 'char';
      maxLength: number;
      lowercase: boolean;
    };
    image?: {
      resize: [number, number];
      normalize: boolean;
      augmentation: string[];
    };
  };
  storage: {
    type: 'memory' | 'disk' | 'vector-db';
    index: {
      type: 'flat' | 'hnsw' | 'ivf';
      metric: 'cosine' | 'euclidean' | 'dot';
    };
  };
}

class EmbeddingsManager {
  async generateEmbeddings(
    data: any[],
    config: EmbeddingsConfig
  ) {
    // Preprocess input data
    const processed = await this.preprocess(
      data,
      config.preprocessing
    );
    
    // Generate embeddings
    const embeddings = await this.embed(
      processed,
      config.model
    );
    
    // Store embeddings
    await this.store(
      embeddings,
      config.storage
    );
    
    return embeddings;
  }
}`}
      />
    </section>
  );
}

export default EmbeddingsTools;