import React from 'react';
import CodeBlock from '../../CodeBlock';

function PineconeTools() {
  return (
    <section id="pinecone-tools">
      <h2>Pinecone Tools</h2>
      <p>
        Advanced tools for managing vector databases with Pinecone, including index management,
        upsert operations, and similarity search.
      </p>

      <h3>Index Management</h3>
      <CodeBlock
        language="typescript"
        code={`interface PineconeConfig {
  index: {
    name: string;
    dimension: number;
    metric: 'cosine' | 'euclidean' | 'dotproduct';
    pods: number;
    replicas: number;
    shards: number;
  };
  metadata: {
    indexed: string[];
    stored: string[];
  };
  optimization: {
    indexType: 'approximated' | 'exact';
    parameters: {
      ef: number;
      maxConnections: number;
    };
  };
}

class PineconeManager {
  async createIndex(config: PineconeConfig) {
    // Initialize Pinecone client
    const pinecone = await this.initializeClient();
    
    // Create index with configuration
    await pinecone.createIndex({
      name: config.index.name,
      dimension: config.index.dimension,
      metric: config.index.metric,
      pods: config.index.pods,
      replicas: config.index.replicas,
      shards: config.index.shards,
      metadata: {
        indexed: config.metadata.indexed,
        stored: config.metadata.stored
      }
    });
    
    // Apply optimization settings
    await this.optimizeIndex(
      config.index.name,
      config.optimization
    );
    
    return pinecone.Index(config.index.name);
  }`}
      />

      <h3>Vector Operations</h3>
      <CodeBlock
        language="typescript"
        code={`interface VectorOperations {
  upsert: {
    batchSize: number;
    namespace?: string;
    metadata?: Record<string, any>;
  };
  query: {
    topK: number;
    includeMetadata: boolean;
    includeValues: boolean;
    filter?: Record<string, any>;
  };
  fetch: {
    ids: string[];
    namespace?: string;
  };
  delete: {
    ids?: string[];
    filter?: Record<string, any>;
    namespace?: string;
    deleteAll?: boolean;
  };
}

class VectorOperator {
  async upsertVectors(
    vectors: Float32Array[],
    config: VectorOperations['upsert']
  ) {
    // Split vectors into batches
    const batches = this.createBatches(
      vectors,
      config.batchSize
    );
    
    // Upsert each batch
    const results = await Promise.all(
      batches.map(batch =>
        this.index.upsert({
          vectors: batch,
          namespace: config.namespace,
          metadata: config.metadata
        })
      )
    );
    
    return this.aggregateResults(results);
  }

  async querySimilar(
    vector: Float32Array,
    config: VectorOperations['query']
  ) {
    return this.index.query({
      vector,
      topK: config.topK,
      includeMetadata: config.includeMetadata,
      includeValues: config.includeValues,
      filter: config.filter
    });
  }
}`}
      />

      <h3>Metadata Management</h3>
      <CodeBlock
        language="typescript"
        code={`interface MetadataConfig {
  schema: {
    fields: {
      name: string;
      type: 'string' | 'number' | 'boolean' | 'array';
      indexed: boolean;
    }[];
  };
  validation: {
    enabled: boolean;
    strict: boolean;
    coerce: boolean;
  };
  updates: {
    merge: boolean;
    overwrite: boolean;
    upsert: boolean;
  };
}

class MetadataManager {
  async updateMetadata(
    ids: string[],
    metadata: Record<string, any>,
    config: MetadataConfig
  ) {
    // Validate metadata against schema
    if (config.validation.enabled) {
      await this.validateMetadata(
        metadata,
        config.schema
      );
    }
    
    // Apply metadata updates
    return this.index.update({
      ids,
      metadata,
      ...config.updates
    });
  }
}`}
      />

      <div className="bg-gray-800 rounded-lg p-6 my-8">
        <h4 className="text-lg font-semibold mb-4">Best Practices</h4>
        <ul className="space-y-2">
          <li>Use appropriate batch sizes for upsert operations</li>
          <li>Implement retry logic for network failures</li>
          <li>Monitor index metrics and scale as needed</li>
          <li>Optimize metadata schema for query performance</li>
          <li>Implement proper error handling for all operations</li>
        </ul>
      </div>
    </section>
  );
}

export default PineconeTools;