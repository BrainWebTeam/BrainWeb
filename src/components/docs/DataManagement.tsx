import React from 'react';
import CodeBlock from '../CodeBlock';

function DataManagement() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1>Data Management</h1>

      <h2>State Management</h2>
      <p>
        The network implements a robust state management system to handle complex data relationships and updates:
      </p>

      <CodeBlock
        language="typescript"
        code={`interface NetworkState {
  nodes: Node[];
  connections: Connection[];
  selectedNode: Node | null;
  hoveredNode: Node | null;
  userProgress: UserProgress;
  viewState: ViewState;
}

const useNetworkState = () => {
  const [state, dispatch] = useReducer(networkReducer, initialState);

  const updateNodeState = useCallback((nodeId: string, changes: Partial<Node>) => {
    dispatch({ type: 'UPDATE_NODE', payload: { nodeId, changes } });
  }, []);

  const activateNode = useCallback((nodeId: string) => {
    dispatch({ type: 'ACTIVATE_NODE', payload: { nodeId } });
  }, []);

  return {
    state,
    updateNodeState,
    activateNode
  };
};`}
      />

      <h2>Caching Strategy</h2>
      <p>
        The system implements efficient caching to optimize performance and reduce data fetching:
      </p>

      <CodeBlock
        language="typescript"
        code={`class ContentCache {
  private cache: Map<string, NodeContent>;
  private maxSize: number;
  private expiryTime: number;

  constructor(maxSize = 100, expiryTime = 3600000) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.expiryTime = expiryTime;
  }

  async getContent(nodeId: string): Promise<NodeContent> {
    if (this.cache.has(nodeId)) {
      const entry = this.cache.get(nodeId);
      if (!this.isExpired(entry)) {
        return entry.content;
      }
    }

    const content = await this.fetchContent(nodeId);
    this.cache.set(nodeId, {
      content,
      timestamp: Date.now()
    });

    this.pruneCache();
    return content;
  }

  private pruneCache() {
    if (this.cache.size > this.maxSize) {
      const oldestKey = Array.from(this.cache.keys())[0];
      this.cache.delete(oldestKey);
    }
  }
}`}
      />

      <h2>Data Synchronization</h2>
      <p>
        The network maintains data consistency across components through a robust synchronization system:
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Sync Features</h3>
          <ul>
            <li>Real-time updates</li>
            <li>Conflict resolution</li>
            <li>Optimistic updates</li>
            <li>Retry mechanisms</li>
            <li>Error recovery</li>
          </ul>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Data Flow</h3>
          <ul>
            <li>Unidirectional flow</li>
            <li>Event propagation</li>
            <li>State immutability</li>
            <li>Change detection</li>
            <li>Update batching</li>
          </ul>
        </div>
      </div>

      <h2>Error Handling</h2>
      <CodeBlock
        language="typescript"
        code={`class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    logError({
      error,
      component: info.componentStack,
      timestamp: Date.now()
    });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <button onClick={this.resetError}>
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}`}
      />

      <h2>Data Persistence</h2>
      <p>
        The system implements multiple strategies for data persistence:
      </p>

      <CodeBlock
        language="typescript"
        code={`class StorageManager {
  private storage: Storage;
  private prefix: string;

  constructor(storage: Storage = localStorage) {
    this.storage = storage;
    this.prefix = 'network_';
  }

  saveProgress(userId: string, progress: UserProgress) {
    const key = this.getKey('progress', userId);
    const data = JSON.stringify(progress);
    this.storage.setItem(key, data);
  }

  loadProgress(userId: string): UserProgress | null {
    const key = this.getKey('progress', userId);
    const data = this.storage.getItem(key);
    return data ? JSON.parse(data) : null;
  }

  private getKey(...parts: string[]): string {
    return [this.prefix, ...parts].join('_');
  }
}`}
      />

      <h2>Performance Optimization</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Memory Management</h3>
          <ul>
            <li>Cache size limits</li>
            <li>Resource cleanup</li>
            <li>Memory pooling</li>
            <li>Garbage collection</li>
            <li>Reference tracking</li>
          </ul>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Data Loading</h3>
          <ul>
            <li>Lazy loading</li>
            <li>Progressive loading</li>
            <li>Prefetching</li>
            <li>Data chunking</li>
            <li>Priority queues</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default DataManagement;