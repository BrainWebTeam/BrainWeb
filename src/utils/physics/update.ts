import { PhysicsNode } from '../../types/physics';
import { PHYSICS_CONFIG } from '../../config/physics';
import { calculateForces } from './forces';

export const updatePhysics = (nodes: PhysicsNode[], deltaTime: number): PhysicsNode[] => {
  const dt = Math.min(deltaTime, 32) * 0.001;
  const draggedNode = nodes.find(n => n.isDragging);
  
  return nodes.map(node => {
    if (node.isDragging || !node.originalX || !node.originalY) return node;

    const forces = calculateForces(node, draggedNode, dt);
    
    // Update velocity with smooth acceleration
    let newVelX = node.velocityX + forces.x;
    let newVelY = node.velocityY + forces.y;

    // Apply damping
    newVelX *= PHYSICS_CONFIG.SPRING_DAMPING;
    newVelY *= PHYSICS_CONFIG.SPRING_DAMPING;

    // Limit velocity
    const speed = Math.sqrt(newVelX * newVelX + newVelY * newVelY);
    if (speed > PHYSICS_CONFIG.MAX_VELOCITY) {
      const scale = PHYSICS_CONFIG.MAX_VELOCITY / speed;
      newVelX *= scale;
      newVelY *= scale;
    }

    // Update position with velocity
    const newX = node.x + newVelX * dt;
    const newY = node.y + newVelY * dt;

    return {
      ...node,
      x: Number.isFinite(newX) ? newX : node.x,
      y: Number.isFinite(newY) ? newY : node.y,
      velocityX: Number.isFinite(newVelX) ? newVelX : 0,
      velocityY: Number.isFinite(newVelY) ? newVelY : 0
    };
  });
};