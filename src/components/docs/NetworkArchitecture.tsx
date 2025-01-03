import React from 'react';
import CodeBlock from '../CodeBlock';

function NetworkArchitecture() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1>Network Architecture</h1>
      
      <h2>Layer Organization</h2>
      <p>
        The AI Learning Network is structured in concentric layers, each representing a different level of knowledge depth and specialization:
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Layer Structure</h3>
          <ul>
            <li><strong>Core Layer:</strong> Fundamental AI concepts</li>
            <li><strong>Primary Layer:</strong> Major AI branches</li>
            <li><strong>Secondary Layer:</strong> Specific techniques</li>
            <li><strong>Tertiary Layer:</strong> Advanced applications</li>
            <li><strong>Quaternary Layer:</strong> Specialized topics</li>
          </ul>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Connection Types</h3>
          <ul>
            <li><strong>Radial:</strong> Core to outer layers</li>
            <li><strong>Circular:</strong> Within same layer</li>
            <li><strong>Cross-layer:</strong> Between adjacent layers</li>
            <li><strong>Dynamic:</strong> Based on node state</li>
          </ul>
        </div>
      </div>

      <h2>Node System</h2>
      <CodeBlock
        language="typescript"
        code={`interface Node {
  id: string;
  type: 'center' | 'primary' | 'secondary' | 'tertiary' | 'quaternary';
  x: number;
  y: number;
  activated?: boolean;
  available?: boolean;
}

interface Connection {
  id: string;
  sourceId: string;
  targetId: string;
  strength: number;
}`}
      />

      <h2>Physics Implementation</h2>
      <p>
        The network uses a custom physics engine for natural movement and interactions:
      </p>

      <CodeBlock
        language="typescript"
        code={`interface PhysicsNode extends Node {
  velocity: Vector2D;
  force: Vector2D;
  mass: number;
  damping: number;
  fixed: boolean;
}

class PhysicsEngine {
  private nodes: Map<string, PhysicsNode>;
  private springs: Spring[];

  update(deltaTime: number) {
    this.applyForces();
    this.updatePositions(deltaTime);
    this.applyConstraints();
  }

  private applyForces() {
    // Apply spring forces between connected nodes
    // Apply repulsion forces between all nodes
    // Apply return forces to original positions
  }

  private updatePositions(dt: number) {
    // Update velocities based on forces
    // Update positions based on velocities
    // Apply damping
  }
}`}
      />

      <h2>State Management</h2>
      <p>
        Node states are managed through a sophisticated system that handles activation and availability:
      </p>

      <CodeBlock
        language="typescript"
        code={`const updateNodeState = (nodeId: string) => {
  setNodes(currentNodes => {
    const updatedNodes = currentNodes.map(node => {
      if (node.id === nodeId) {
        return { ...node, activated: true, available: false };
      }
      
      // Update connected nodes
      const isConnected = connections.some(conn => 
        (conn.sourceId === nodeId && conn.targetId === node.id) ||
        (conn.targetId === nodeId && conn.sourceId === node.id)
      );
      
      if (isConnected && !node.activated) {
        return { ...node, available: true };
      }
      
      return node;
    });
    
    return updatedNodes;
  });
};`}
      />

      <h2>Rendering System</h2>
      <p>
        The network uses SVG for high-performance rendering with dynamic updates:
      </p>

      <CodeBlock
        language="typescript"
        code={`const NetworkVisual: React.FC = () => {
  const [nodes, setNodes] = useState<Node[]>(createNodes());
  const connections = createConnections(nodes);
  const { scale, onWheel } = useZoom();
  const { position, handlers } = usePan();

  return (
    <svg 
      viewBox="0 0 800 800"
      onWheel={onWheel}
      {...handlers}
    >
      <g transform={\`translate(\${position.x},\${position.y}) scale(\${scale})\`}>
        <ConnectionLayer connections={connections} nodes={nodes} />
        <NodeLayer nodes={nodes} />
      </g>
    </svg>
  );
};`}
      />

      <h2>Performance Optimizations</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Rendering Optimizations</h3>
          <ul>
            <li>SVG layer management</li>
            <li>Component memoization</li>
            <li>Efficient updates</li>
            <li>Batch processing</li>
            <li>Hardware acceleration</li>
          </ul>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Physics Optimizations</h3>
          <ul>
            <li>Spatial partitioning</li>
            <li>Force capping</li>
            <li>Delta time smoothing</li>
            <li>Collision optimization</li>
            <li>Update throttling</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default NetworkArchitecture;