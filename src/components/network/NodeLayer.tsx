import React, { useMemo } from 'react';
import Node from './Node';
import NodeLabel from './NodeLabel';
import { Node as NodeType } from '../../types/network';
import { NODE_LABELS } from '../../utils/network/constants';

interface NodeLayerProps {
  nodes: NodeType[];
  scale: number;
  time: number;
  onNodeClick?: (id: string) => void;
}

const NodeLayer: React.FC<NodeLayerProps> = ({ 
  nodes, 
  scale,
  onNodeClick 
}) => {
  // Memoize node sizes
  const nodeSizes = useMemo(() => ({
    center: 12,
    primary: 8,
    secondary: 6,
    tertiary: 4,
    quaternary: 3
  }), []);

  // Memoize active and available nodes
  const { activeNodes, availableNodes } = useMemo(() => {
    const active = nodes.filter(n => n.activated);
    const available = nodes.filter(n => n.available);
    return { activeNodes: active, availableNodes: available };
  }, [nodes]);

  // Get node label
  const getNodeLabel = (node: NodeType): string => {
    const typeLabels = NODE_LABELS[node.type];
    const index = parseInt(node.id.split('-')[1] || '0');
    return typeLabels[index] || '';
  };

  return (
    <g className="nodes-layer">
      {/* Render inactive nodes with minimal effects */}
      {nodes.map(node => {
        if (node.activated || node.available) return null;
        return (
          <Node
            key={node.id}
            x={node.x}
            y={node.y}
            size={nodeSizes[node.type] / scale}
            activated={false}
            available={false}
          />
        );
      })}

      {/* Render available nodes */}
      {availableNodes.map(node => (
        <React.Fragment key={node.id}>
          <Node
            x={node.x}
            y={node.y}
            size={nodeSizes[node.type] / scale}
            activated={false}
            available={true}
            onClick={() => onNodeClick?.(node.id)}
          />
          <NodeLabel
            x={node.x}
            y={node.y}
            label={getNodeLabel(node)}
            available={true}
          />
        </React.Fragment>
      ))}

      {/* Render active nodes */}
      {activeNodes.map(node => (
        <React.Fragment key={node.id}>
          <Node
            x={node.x}
            y={node.y}
            size={nodeSizes[node.type] / scale}
            activated={true}
            onClick={() => onNodeClick?.(node.id)}
          />
          <NodeLabel
            x={node.x}
            y={node.y}
            label={getNodeLabel(node)}
            activated={true}
          />
        </React.Fragment>
      ))}
    </g>
  );
};

export default React.memo(NodeLayer);