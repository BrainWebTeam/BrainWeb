import { Node } from '../../types/network';
import { CENTER, RADII, NODE_COUNTS } from './constants';

const createLayerNodes = (
  count: number,
  radius: number,
  type: Node['type'],
  angleOffset = 0
): Node[] => {
  return Array.from({ length: count }, (_, i) => {
    const angle = (i * 2 * Math.PI / count) + angleOffset;
    return {
      id: `${type}-${i}`,
      x: CENTER.x + Math.cos(angle) * radius,
      y: CENTER.y + Math.sin(angle) * radius,
      type
    };
  });
};

export const createNodes = (): Node[] => {
  return [
    // Center node
    {
      id: 'center',
      x: CENTER.x,
      y: CENTER.y,
      type: 'center'
    },
    // Layer nodes with perfect circular placement
    ...createLayerNodes(NODE_COUNTS.PRIMARY, RADII.PRIMARY, 'primary'),
    ...createLayerNodes(NODE_COUNTS.SECONDARY, RADII.SECONDARY, 'secondary', Math.PI / NODE_COUNTS.SECONDARY),
    ...createLayerNodes(NODE_COUNTS.TERTIARY, RADII.TERTIARY, 'tertiary', Math.PI / NODE_COUNTS.TERTIARY),
    ...createLayerNodes(NODE_COUNTS.QUATERNARY, RADII.QUATERNARY, 'quaternary', Math.PI / NODE_COUNTS.QUATERNARY)
  ];
};