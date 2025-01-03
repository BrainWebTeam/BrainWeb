import { NodeContent } from '../../types/content';
import { Node } from '../../types/network';
import { nodeContentMap } from '../../data/nodeContent';

export const getNodeContent = (node: Node): NodeContent | null => {
  return nodeContentMap[node.id] || null;
};