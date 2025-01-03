import { Node } from '../../types/network';

export const calculateStrength = (node1: Node, node2: Node): number => {
  const dx = node2.x - node1.x;
  const dy = node2.y - node1.y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  
  // Base strength on node types
  let baseStrength = 1;
  if (node1.type === 'center' || node2.type === 'center') {
    baseStrength = 1;
  } else if (node1.type === 'primary' || node2.type === 'primary') {
    baseStrength = 0.8;
  } else if (node1.type === 'secondary' || node2.type === 'secondary') {
    baseStrength = 0.6;
  } else {
    baseStrength = 0.4;
  }
  
  // Decrease strength with distance
  return baseStrength * Math.max(0.2, 1 - distance / 500);
};