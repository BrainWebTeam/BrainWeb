import React from 'react';
import CodeBlock from '../CodeBlock';

function AdvancedFeatures() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1>Advanced Features</h1>

      <h2>Interactive Network Visualization</h2>
      <p>
        The network visualization is built using advanced SVG techniques and physics-based animations to create a natural, responsive interface.
      </p>

      <h3>Physics Engine</h3>
      <CodeBlock
        language="typescript"
        code={`const PHYSICS_CONFIG = {
  // Spring physics
  SPRING_DAMPING: 0.92,
  SPRING_STIFFNESS: 0.03,
  
  // Node connections
  CONNECTION_STRENGTH: 0.3,
  CONNECTION_RADIUS: 250,
  
  // Movement
  RETURN_FORCE: 0.04,
  RETURN_DAMPING: 0.95,
  
  // Animation
  MIN_VELOCITY: 0.01,
  MAX_VELOCITY: 20,
  MAX_DISPLACEMENT: 400,
  
  // Interaction
  DRAG_INFLUENCE: 0.7,
  WOBBLE_AMOUNT: 0.3,
  WOBBLE_SPEED: 1.5
};`}
      />

      <h3>Node Management</h3>
      <p>
        Nodes are organized in concentric layers with dynamic connections and state management:
      </p>

      <CodeBlock
        language="typescript"
        code={`const createLayerNodes = (
  count: number,
  radius: number,
  type: Node['type'],
  angleOffset = 0
): Node[] => {
  return Array.from({ length: count }, (_, i) => {
    const angle = (i * 2 * Math.PI / count) + angleOffset;
    return {
      id: \`\${type}-\${i}\`,
      x: CENTER.x + Math.cos(angle) * radius,
      y: CENTER.y + Math.sin(angle) * radius,
      type
    };
  });
};`}
      />

      <h2>Knowledge Panel System</h2>
      <p>
        The knowledge panel provides detailed information about AI concepts with a rich, interactive interface:
      </p>

      <CodeBlock
        language="typescript"
        code={`interface NodeContent {
  title: string;
  description: string;
  concepts: string[];
  examples: {
    language: string;
    code: string;
    description: string;
  }[];
  resources: {
    title: string;
    description: string;
    url: string;
  }[];
  prerequisites?: string[];
  relatedTopics?: string[];
}`}
      />

      <h2>Animation System</h2>
      <p>
        The network features sophisticated animations powered by Framer Motion and custom physics calculations:
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Animation Features</h3>
          <ul>
            <li>Smooth node transitions</li>
            <li>Connection animations</li>
            <li>Physics-based movement</li>
            <li>Interactive feedback</li>
            <li>Particle effects</li>
          </ul>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3>Physics Features</h3>
          <ul>
            <li>Spring-based connections</li>
            <li>Force-directed layout</li>
            <li>Damping and stability</li>
            <li>Collision detection</li>
            <li>Smooth interpolation</li>
          </ul>
        </div>
      </div>

      <h2>Connection System</h2>
      <p>
        Connections between nodes are managed through a sophisticated system that handles both visual and logical relationships:
      </p>

      <CodeBlock
        language="typescript"
        code={`const createConnections = (nodes: Node[]): Connection[] => {
  const connections: Connection[] = [];
  const center = nodes.find(n => n.type === 'center')!;
  
  // Filter nodes by type
  const primaryNodes = nodes.filter(n => n.type === 'primary');
  const secondaryNodes = nodes.filter(n => n.type === 'secondary');
  
  // Connect primary nodes to center
  primaryNodes.forEach(node => {
    connections.push({
      id: \`center-\${node.id}\`,
      sourceId: center.id,
      targetId: node.id,
      strength: CONNECTION_STRENGTHS.CENTER
    });
  });

  // Create circular connections
  primaryNodes.forEach((node, i) => {
    const next = primaryNodes[(i + 1) % primaryNodes.length];
    connections.push({
      id: \`primary-\${node.id}-\${next.id}\`,
      sourceId: node.id,
      targetId: next.id,
      strength: CONNECTION_STRENGTHS.PRIMARY
    });
  });

  return connections;
};`}
      />

      <h2>Interaction Handling</h2>
      <p>
        The network provides rich interaction capabilities through custom hooks and event handlers:
      </p>

      <CodeBlock
        language="typescript"
        code={`const useInteraction = () => {
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null);

  const handleNodeClick = useCallback((nodeId: string) => {
    const node = nodes.find(n => n.id === nodeId);
    if (node?.available || node?.activated) {
      setSelectedNode(node);
      updateNodeState(nodeId);
    }
  }, [nodes]);

  const handleNodeHover = useCallback((nodeId: string | null) => {
    const node = nodeId ? nodes.find(n => n.id === nodeId) : null;
    setHoveredNode(node);
  }, [nodes]);

  return {
    selectedNode,
    hoveredNode,
    handleNodeClick,
    handleNodeHover
  };
};`}
      />

      <h2>Performance Optimization</h2>
      <p>
        The network implements several optimization techniques to ensure smooth performance:
      </p>

      <ul>
        <li>Request animation frame management</li>
        <li>Efficient SVG rendering</li>
        <li>Memoized components</li>
        <li>Batched state updates</li>
        <li>Virtualized rendering for large datasets</li>
      </ul>

      <div className="bg-gray-800 rounded-lg p-6 my-8">
        <h3>Performance Tips</h3>
        <ul>
          <li>Use mouse wheel to zoom in/out</li>
          <li>Drag nodes to explore connections</li>
          <li>Click nodes to view detailed information</li>
          <li>Double-click to reset the view</li>
          <li>Use the search feature for quick navigation</li>
        </ul>
      </div>
    </div>
  );
}

export default AdvancedFeatures;