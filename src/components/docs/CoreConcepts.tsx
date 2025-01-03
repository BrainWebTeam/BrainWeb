import React from 'react';
import CodeBlock from '../CodeBlock';

function CoreConcepts() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1>Core Concepts</h1>

      <h2>Network Architecture</h2>
      <p>
        The learning network is organized in concentric layers, each representing different levels of AI knowledge:
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
          <h3>Node Types</h3>
          <ul>
            <li><strong>Concept Nodes:</strong> Core knowledge units</li>
            <li><strong>Connection Nodes:</strong> Relationship indicators</li>
            <li><strong>Available Nodes:</strong> Unlocked content</li>
            <li><strong>Locked Nodes:</strong> Future content</li>
          </ul>
        </div>
      </div>

      <h2>Implementation Details</h2>
      
      <h3>Node Management</h3>
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

const createNodes = (): Node[] => {
  return [
    // Center node
    {
      id: 'center',
      type: 'center',
      x: CENTER.x,
      y: CENTER.y
    },
    // Layer nodes
    ...createLayerNodes(NODE_COUNTS.PRIMARY, RADII.PRIMARY, 'primary'),
    ...createLayerNodes(NODE_COUNTS.SECONDARY, RADII.SECONDARY, 'secondary'),
    // Additional layers...
  ];
};`}
      />

      <h3>Physics System</h3>
      <CodeBlock
        language="typescript"
        code={`const PHYSICS_CONFIG = {
  SPRING_DAMPING: 0.92,
  SPRING_STIFFNESS: 0.03,
  CONNECTION_STRENGTH: 0.3,
  CONNECTION_RADIUS: 250,
  RETURN_FORCE: 0.04,
  RETURN_DAMPING: 0.95,
  MAX_VELOCITY: 20,
  WOBBLE_AMOUNT: 0.3
};

const updatePhysics = (nodes: PhysicsNode[], deltaTime: number) => {
  const dt = Math.min(deltaTime, 32) * 0.001;
  
  return nodes.map(node => {
    if (node.isDragging) return node;
    
    // Calculate forces
    const forces = calculateForces(node);
    
    // Update velocity and position
    const newVelocity = updateVelocity(node, forces, dt);
    const newPosition = updatePosition(node, newVelocity, dt);
    
    return { ...node, ...newPosition, ...newVelocity };
  });
};`}
      />

      <h2>Knowledge Management</h2>
      <p>
        Each node in the network contains structured information:
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

      <h2>Visual Effects</h2>
      <p>
        The network features various visual effects to enhance user experience:
      </p>

      <ul>
        <li>Dynamic node pulsing for available content</li>
        <li>Smooth transitions for state changes</li>
        <li>Force-directed layout for natural positioning</li>
        <li>Interactive zoom and pan controls</li>
        <li>Particle effects for engagement</li>
      </ul>
    </div>
  );
}

export default CoreConcepts;