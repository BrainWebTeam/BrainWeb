import { PhysicsNode, Vector2D } from './types';
import { PHYSICS } from './constants';
import { vector } from './vector';

export const integration = {
  verlet: (node: PhysicsNode, dt: number): void => {
    if (node.fixed || node.isDragging) return;
    
    // Update velocity using Verlet integration
    const acceleration = vector.divide(node.force, node.mass);
    const dtSquared = dt * dt;
    
    const newPosition = {
      x: node.position.x + node.velocity.x * dt + 0.5 * acceleration.x * dtSquared,
      y: node.position.y + node.velocity.y * dt + 0.5 * acceleration.y * dtSquared,
    };
    
    // Update velocity
    const newVelocity = {
      x: (newPosition.x - node.position.x) / dt,
      y: (newPosition.y - node.position.y) / dt,
    };
    
    // Apply damping
    newVelocity.x *= node.damping;
    newVelocity.y *= node.damping;
    
    // Clamp velocity
    const speed = vector.magnitude(newVelocity);
    if (speed > PHYSICS.VELOCITY.MAX) {
      const scale = PHYSICS.VELOCITY.MAX / speed;
      newVelocity.x *= scale;
      newVelocity.y *= scale;
    }
    
    // Update node
    node.position = newPosition;
    node.velocity = newVelocity;
  },
  
  eulerSemi: (node: PhysicsNode, dt: number): void => {
    if (node.fixed || node.isDragging) return;
    
    // Update velocity first (semi-implicit Euler)
    const acceleration = vector.divide(node.force, node.mass);
    node.velocity = vector.add(node.velocity, vector.multiply(acceleration, dt));
    
    // Apply damping
    node.velocity = vector.multiply(node.velocity, node.damping);
    
    // Clamp velocity
    node.velocity = vector.clamp(node.velocity, -PHYSICS.VELOCITY.MAX, PHYSICS.VELOCITY.MAX);
    
    // Update position using new velocity
    node.position = vector.add(node.position, vector.multiply(node.velocity, dt));
  },
};