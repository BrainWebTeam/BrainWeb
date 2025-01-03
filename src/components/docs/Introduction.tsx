import React from 'react';
import { Code } from 'lucide-react';
import CodeBlock from '../CodeBlock';

function Introduction() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1>AI Learning Network</h1>
      
      <p className="lead text-xl text-gray-300">
        An interactive visualization platform for exploring artificial intelligence concepts through an interconnected knowledge network.
      </p>

      <h2>Overview</h2>
      <p>
        The AI Learning Network provides an intuitive way to explore and understand AI concepts through a visual, interactive interface.
        The network organizes knowledge in concentric layers, from fundamental concepts at the core to specialized topics in outer layers.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-blue-400 flex items-center gap-2 mb-4">
            <Code className="w-5 h-5" />
            Key Features
          </h3>
          <ul className="space-y-2">
            <li>Interactive network visualization</li>
            <li>Progressive learning paths</li>
            <li>Detailed concept explanations</li>
            <li>Code examples and resources</li>
            <li>Visual knowledge mapping</li>
          </ul>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-blue-400 flex items-center gap-2 mb-4">
            <Code className="w-5 h-5" />
            Network Structure
          </h3>
          <ul className="space-y-2">
            <li>Core: Fundamental AI concepts</li>
            <li>Primary: Major AI branches</li>
            <li>Secondary: Specific techniques</li>
            <li>Tertiary: Advanced applications</li>
            <li>Quaternary: Specialized topics</li>
          </ul>
        </div>
      </div>

      <h2>Technical Implementation</h2>
      <p>
        The network is built using React and TypeScript, featuring advanced visualization techniques and physics-based animations.
        Here's an example of the core network visualization component:
      </p>

      <CodeBlock
        language="typescript"
        code={`import React, { useState, useCallback } from 'react';
import { Node, Connection } from '../types/network';
import { usePhysicsSimulation } from '../hooks/usePhysicsSimulation';

const NetworkVisual: React.FC = () => {
  const [nodes, setNodes] = useState<Node[]>(createInitialNodes());
  const connections = createConnections(nodes);

  const handleNodeClick = useCallback((nodeId: string) => {
    setNodes(currentNodes => 
      currentNodes.map(node => ({
        ...node,
        activated: node.id === nodeId ? true : node.activated
      }))
    );
  }, []);

  return (
    <svg className="network-visual" viewBox="0 0 800 800">
      <ConnectionLayer connections={connections} />
      <NodeLayer 
        nodes={nodes}
        onNodeClick={handleNodeClick}
      />
    </svg>
  );
};`}
      />

      <h2>Physics Engine</h2>
      <p>
        The network uses a custom physics engine for natural and responsive interactions:
      </p>

      <CodeBlock
        language="typescript"
        code={`interface PhysicsNode {
  position: Vector2D;
  velocity: Vector2D;
  force: Vector2D;
  mass: number;
  damping: number;
}

class PhysicsEngine {
  private nodes: Map<string, PhysicsNode>;
  private springs: Spring[];

  update(deltaTime: number) {
    // Apply forces
    this.applySpringForces();
    this.applyDamping();

    // Update positions
    this.nodes.forEach(node => {
      if (!node.fixed) {
        node.position.x += node.velocity.x * deltaTime;
        node.position.y += node.velocity.y * deltaTime;
      }
    });
  }
}`}
      />

      <h2>Getting Started</h2>
      <p>
        To begin exploring the AI Learning Network:
      </p>

      <ol className="list-decimal pl-6 space-y-2">
        <li>Start at the central AI Core node</li>
        <li>Click on available (pulsing) nodes to explore new concepts</li>
        <li>Read through concept descriptions and examples</li>
        <li>Follow suggested learning paths through connected nodes</li>
        <li>Access additional resources for deeper learning</li>
      </ol>

      <div className="bg-gray-800 rounded-lg p-6 my-8">
        <h3 className="text-blue-400 mb-4">Navigation Tips</h3>
        <ul className="space-y-2">
          <li>Use mouse wheel to zoom in/out</li>
          <li>Click and drag to pan around the network</li>
          <li>Click on nodes to view detailed information</li>
          <li>Follow the suggested paths for structured learning</li>
        </ul>
      </div>
    </div>
  );
}

export default Introduction;