import { PhysicsNode, Vector2D } from './types';
import { vector } from './vector';
import { PHYSICS } from './constants';

export const constraints = {
  boundingBox: (node: PhysicsNode, bounds: { min: Vector2D; max: Vector2D }): void => {
    // X-axis constraints
    if (node.position.x < bounds.min.x) {
      node.position.x = bounds.min.x;
      node.velocity.x *= -0.5; // Bounce with energy loss
    } else if (node.position.x > bounds.max.x) {
      node.position.x = bounds.max.x;
      node.velocity.x *= -0.5;
    }
    
    // Y-axis constraints
    if (node.position.y < bounds.min.y) {
      node.position.y = bounds.min.y;
      node.velocity.y *= -0.5;
    } else if (node.position.y > bounds.max.y) {
      node.position.y = bounds.max.y;
      node.velocity.y *= -0.5;
    }
  },
  
  distance: (nodeA: PhysicsNode, nodeB: PhysicsNode, minDist: number, maxDist: number): void => {
    const displacement = vector.subtract(nodeB.position, nodeA.position);
    const distance = vector.magnitude(displacement);
    
    if (distance === 0 || (distance >= minDist && distance <= maxDist)) return;
    
    const normalized = vector.normalize(displacement);
    const correction = distance < minDist ? minDist - distance : maxDist - distance;
    const move = vector.multiply(normalized, correction * 0.5);
    
    if (!nodeA.fixed && !nodeA.isDragging) {
      nodeA.position = vector.subtract(nodeA.position, move);
    }
    if (!nodeB.fixed && !nodeB.isDragging) {
      nodeB.position = vector.add(nodeB.position, move);
    }
  },
};