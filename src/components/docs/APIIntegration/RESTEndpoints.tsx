import React from 'react';
import CodeBlock from '../../CodeBlock';

function RESTEndpoints() {
  return (
    <section>
      <h2>REST Endpoints</h2>
      <p>
        Core API endpoints and their usage:
      </p>

      <CodeBlock
        language="typescript"
        code={`// API client setup
const api = createAPIClient({
  baseURL: '/api/v1',
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Endpoint implementations
const endpoints = {
  // Learning paths
  async getLearningPaths() {
    return api.get('/learning-paths');
  },
  
  // Node content
  async getNodeContent(nodeId: string) {
    return api.get(\`/nodes/\${nodeId}/content\`);
  },
  
  // User progress
  async updateProgress(nodeId: string, progress: Progress) {
    return api.post(\`/progress/\${nodeId}\`, progress);
  },
  
  // Analytics
  async trackInteraction(data: InteractionData) {
    return api.post('/analytics/interactions', data);
  }
};`}
      />
    </section>
  );
}

export default RESTEndpoints;