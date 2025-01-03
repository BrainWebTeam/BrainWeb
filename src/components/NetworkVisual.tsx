import React, { useState, useCallback } from 'react';
import ConnectionLayer from './network/ConnectionLayer';
import NodeLayer from './network/NodeLayer';
import KnowledgePanel from './KnowledgePanel';
import { createNodes, createConnections } from '../utils/network';
import { getNodeContent } from '../utils/network/getNodeContent';
import { useZoom } from '../hooks/useZoom';
import { usePan } from '../hooks/usePan';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { Node } from '../types/network';

const NetworkVisual: React.FC = () => {
  const [time, setTime] = useState(0);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [nodes, setNodes] = useState<Node[]>(() => {
    const initialNodes = createNodes();
    return initialNodes.map(node => ({
      ...node,
      activated: node.type === 'center',
      available: node.type === 'primary'
    }));
  });

  const connections = createConnections(nodes);
  const { scale, onWheel } = useZoom();
  const { position, handlers } = usePan();
  const [cursor, setCursor] = useState('grab');

  useAnimationFrame((deltaTime) => {
    setTime(prev => prev + deltaTime * 0.0003);
  });

  const handleNodeClick = useCallback((nodeId: string) => {
    const clickedNode = nodes.find(n => n.id === nodeId);
    if (!clickedNode?.available && !clickedNode?.activated) return;

    setSelectedNode(clickedNode);
    
    if (clickedNode.available) {
      setNodes(currentNodes => 
        currentNodes.map(node => {
          if (node.id === nodeId) {
            return { ...node, activated: true, available: false };
          }

          // Only update availability for non-activated nodes
          if (!node.activated) {
            const isConnected = connections.some(conn => 
              (conn.sourceId === nodeId && conn.targetId === node.id) ||
              (conn.targetId === nodeId && conn.sourceId === node.id)
            );
            if (isConnected) {
              return { ...node, available: true };
            }
          }

          return node;
        })
      );
    }
  }, [connections, nodes]);

  const selectedNodeContent = selectedNode ? getNodeContent(selectedNode) : null;

  return (
    <div className="relative w-full h-full overflow-hidden bg-[#050505]">
      <div 
        className="w-full h-full"
        style={{ cursor }}
        onMouseDown={() => setCursor('grabbing')}
        onMouseUp={() => setCursor('grab')}
        onMouseLeave={() => setCursor('grab')}
        {...handlers}
      >
        <svg 
          viewBox="0 0 800 800"
          className="w-full touch-none"
          style={{ maxHeight: '80vh' }}
          onWheel={onWheel}
        >
          <g style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            transformOrigin: 'center',
          }}>
            <ConnectionLayer 
              connections={connections} 
              nodes={nodes} 
              scale={scale}
            />
            <NodeLayer 
              nodes={nodes} 
              scale={scale}
              time={time}
              onNodeClick={handleNodeClick}
            />
          </g>
        </svg>
      </div>
      <KnowledgePanel 
        node={selectedNode}
        content={selectedNodeContent}
        onClose={() => setSelectedNode(null)}
      />
    </div>
  );
};

export default NetworkVisual;