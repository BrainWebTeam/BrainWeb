import { Node, Connection } from '../../types/network';
import { CONNECTION_STRENGTHS, CENTER } from './constants';

const findNearestInAngle = (node: Node, candidates: Node[], maxAngle: number): Node[] => {
  const nodeAngle = Math.atan2(node.y - CENTER.y, node.x - CENTER.x);
  
  return candidates
    .filter(c => c !== node)
    .filter(c => {
      const candAngle = Math.atan2(c.y - CENTER.y, c.x - CENTER.x);
      let diff = Math.abs(nodeAngle - candAngle);
      diff = Math.min(diff, 2 * Math.PI - diff);
      return diff <= maxAngle;
    })
    .slice(0, 2);
};

export const createConnections = (nodes: Node[]): Connection[] => {
  const connections: Connection[] = [];
  const center = nodes.find(n => n.type === 'center')!;
  
  // Filter nodes by type
  const primaryNodes = nodes.filter(n => n.type === 'primary');
  const secondaryNodes = nodes.filter(n => n.type === 'secondary');
  const tertiaryNodes = nodes.filter(n => n.type === 'tertiary');
  const quaternaryNodes = nodes.filter(n => n.type === 'quaternary');

  // Connect primary nodes to center and neighbors
  primaryNodes.forEach((node, i) => {
    // Connect to center
    connections.push({
      id: `center-${node.id}`,
      sourceId: center.id,
      targetId: node.id,
      strength: CONNECTION_STRENGTHS.CENTER
    });
    
    // Connect to neighbors
    const next = primaryNodes[(i + 1) % primaryNodes.length];
    connections.push({
      id: `primary-${node.id}-${next.id}`,
      sourceId: node.id,
      targetId: next.id,
      strength: CONNECTION_STRENGTHS.PRIMARY
    });
  });

  // Connect other layers with circular connections
  [
    { nodes: secondaryNodes, strength: CONNECTION_STRENGTHS.SECONDARY },
    { nodes: tertiaryNodes, strength: CONNECTION_STRENGTHS.TERTIARY },
    { nodes: quaternaryNodes, strength: CONNECTION_STRENGTHS.QUATERNARY }
  ].forEach(layer => {
    layer.nodes.forEach((node, i) => {
      // Connect to inner layer
      const innerNodes = nodes.filter(n => {
        const typeOrder = ['center', 'primary', 'secondary', 'tertiary', 'quaternary'];
        return typeOrder.indexOf(n.type) === typeOrder.indexOf(node.type) - 1;
      });
      
      // Connect to nearest inner node
      const nearest = findNearestInAngle(node, innerNodes, Math.PI / 8)[0];
      if (nearest) {
        connections.push({
          id: `${node.type}-inner-${node.id}-${nearest.id}`,
          sourceId: node.id,
          targetId: nearest.id,
          strength: layer.strength
        });
      }

      // Connect to neighbors in same layer
      const next = layer.nodes[(i + 1) % layer.nodes.length];
      connections.push({
        id: `${node.type}-${node.id}-${next.id}`,
        sourceId: node.id,
        targetId: next.id,
        strength: layer.strength * 0.8
      });
    });
  });

  return connections;
};