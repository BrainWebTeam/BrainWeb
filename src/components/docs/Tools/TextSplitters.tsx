import React from 'react';
import CodeBlock from '../../CodeBlock';

function TextSplitters() {
  return (
    <section id="text-splitters">
      <h2>Text Splitters</h2>
      <p>
        Advanced tools for splitting text into chunks with support for multiple splitting strategies,
        overlap control, and metadata preservation.
      </p>

      <h3>Character Text Splitter</h3>
      <CodeBlock
        language="typescript"
        code={`interface CharacterSplitterConfig {
  chunkSize: number;
  chunkOverlap: number;
  trimWhitespace: boolean;
  preserveMetadata: boolean;
  lengthFunction: (text: string) => number;
}

class CharacterTextSplitter {
  async splitText(
    text: string,
    config: CharacterSplitterConfig
  ) {
    // Validate configuration
    this.validateConfig(config);
    
    // Split text into chunks
    const chunks = this.createChunks(
      text,
      config.chunkSize,
      config.chunkOverlap
    );
    
    // Process chunks
    return chunks.map(chunk => ({
      text: config.trimWhitespace ? chunk.trim() : chunk,
      metadata: config.preserveMetadata ? this.extractMetadata(chunk) : undefined
    }));
  }
}`}
      />

      <h3>Token Text Splitter</h3>
      <CodeBlock
        language="typescript"
        code={`interface TokenSplitterConfig {
  model: string;
  maxTokens: number;
  overlap: number;
  addSpecialTokens: boolean;
  truncation: {
    strategy: 'head' | 'tail' | 'middle';
    stride: number;
  };
}

class TokenTextSplitter {
  async splitText(
    text: string,
    config: TokenSplitterConfig
  ) {
    // Initialize tokenizer
    const tokenizer = await this.loadTokenizer(config.model);
    
    // Encode text
    const encoding = await tokenizer.encode(text);
    
    // Split into chunks
    const chunks = this.splitTokens(
      encoding,
      config.maxTokens,
      config.overlap
    );
    
    // Decode chunks
    return Promise.all(
      chunks.map(async chunk => ({
        text: await tokenizer.decode(chunk),
        tokens: chunk.length
      }))
    );
  }
}`}
      />

      <h3>Semantic Text Splitter</h3>
      <CodeBlock
        language="typescript"
        code={`interface SemanticSplitterConfig {
  model: string;
  maxChunkSize: number;
  minChunkSize: number;
  strategy: {
    type: 'sentence' | 'paragraph' | 'semantic';
    threshold?: number;
    customBoundaries?: string[];
  };
  cleanup: {
    mergeShortChunks: boolean;
    removeEmpty: boolean;
    normalizeWhitespace: boolean;
  };
}

class SemanticTextSplitter {
  async splitText(
    text: string,
    config: SemanticSplitterConfig
  ) {
    // Load language model
    const model = await this.loadModel(config.model);
    
    // Split into initial chunks
    let chunks = await this.splitByStrategy(
      text,
      config.strategy
    );
    
    // Calculate semantic similarity
    const embeddings = await model.embed(chunks);
    
    // Merge or split chunks based on similarity
    chunks = await this.optimizeChunks(
      chunks,
      embeddings,
      config.maxChunkSize,
      config.minChunkSize
    );
    
    // Apply cleanup
    return this.cleanupChunks(chunks, config.cleanup);
  }
}`}
      />

      <h3>Markdown Text Splitter</h3>
      <CodeBlock
        language="typescript"
        code={`interface MarkdownSplitterConfig {
  headingLevel: number;
  preserveHeaders: boolean;
  preserveLinks: boolean;
  preserveFormatting: boolean;
  maxChunkSize: number;
  customSeparators: string[];
}

class MarkdownTextSplitter {
  async splitText(
    text: string,
    config: MarkdownSplitterConfig
  ) {
    // Parse markdown
    const ast = await this.parseMarkdown(text);
    
    // Split at appropriate boundaries
    const sections = this.splitByHeadings(
      ast,
      config.headingLevel
    );
    
    // Process sections
    return sections.map(section => ({
      text: this.renderSection(section, {
        preserveHeaders: config.preserveHeaders,
        preserveLinks: config.preserveLinks,
        preserveFormatting: config.preserveFormatting
      }),
      metadata: {
        level: section.level,
        title: section.title
      }
    }));
  }
}`}
      />

      <div className="bg-gray-800 rounded-lg p-6 my-8">
        <h4 className="text-lg font-semibold mb-4">Best Practices</h4>
        <ul className="space-y-2">
          <li>Choose appropriate chunk sizes based on model context window</li>
          <li>Use semantic splitting for better context preservation</li>
          <li>Implement proper error handling for edge cases</li>
          <li>Preserve important metadata during splitting</li>
          <li>Consider memory usage for large documents</li>
        </ul>
      </div>
    </section>
  );
}

export default TextSplitters;