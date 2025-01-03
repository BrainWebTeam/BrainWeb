import { PhysicsNode } from '../types/physics';
import { PHYSICS_CONFIG } from '../config/physics';
import { add, subtract, multiply, magnitude, normalize } from './vector';

export const applyReturnForces = (
  nodes: PhysicsNode[],
  config: typeof PHYSICS_CONFIG
) => {
  nodes.forEach(node => {
    if (node.isDragging) return;
    
    const displacement = subtract(node.originalPosition, node.position);
    const distance = magnitude(displacement);
    
    // Apply position constraints first
    if (distance > config.MAX_DISPLACEMENT) {
      const direction = normalize(displacement);
      node.position = add(
        node.originalPosition,
        multiply(direction, -config.MAX_DISPLACEMENT)
      );
      // Strong damping when hitting constraint
      node.velocity = multiply(node.velocity, 0.5);
    }
    
    if (distance < config.POSITION_THRESHOLD) {
      node.position = { ...node.originalPosition };
      node.velocity = { x: 0, y: 0 };
      return;
    }
    
    // Progressive return force that increases with distance
    const forceFactor = Math.pow(distance / 100, 1.5);
    const forceMagnitude = Math.min(
      distance * config.RETURN_FORCE * forceFactor,
      config.MAX_VELOCITY * 2
    );
    
    const returnForce = multiply(normalize(displacement), forceMagnitude);
    
    if (node.justReleased) {
      node.force = multiply(returnForce, 5);
      node.justReleased = false;
    } else {
      node.force = add(node.force, returnForce);
    }
  });
};