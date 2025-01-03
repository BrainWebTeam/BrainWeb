import React from 'react';
import CodeBlock from '../../../CodeBlock';

function WikipediaTools() {
  return (
    <section id="wikipedia-tools">
      <h2>Wikipedia Tools</h2>
      <p>
        Tools for searching and extracting content from Wikipedia.
      </p>

      <CodeBlock
        language="typescript"
        code={`interface WikipediaConfig {
  search: {
    limit: number;
    language: string;
    namespace: number[];
  };
  content: {
    format: 'text' | 'html' | 'wikitext';
    sections: boolean;
    references: boolean;
  };
  cache: {
    enabled: boolean;
    ttl: number;
  };
}

class WikipediaTools {
  async searchArticles(query: string, config: WikipediaConfig) {
    // Initialize Wikipedia client
    const wiki = await this.initClient(config);
    
    // Search for articles
    const results = await wiki.search(query, {
      limit: config.search.limit,
      language: config.search.language
    });
    
    // Get full content for each result
    return Promise.all(
      results.map(result =>
        this.getArticleContent(result.pageid, config.content)
      )
    );
  }
}`}
      />
    </section>
  );
}

export default WikipediaTools;