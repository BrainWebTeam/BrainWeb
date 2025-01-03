import { PhysicsNode } from '../types/physics';
import { PHYSICS_CONFIG as config } from '../config/physics';

const calculateInfluence = (distance: number, maxDistance: number) => {
  if (distance >= maxDistance) return 0;
  const normalized = 1 - distance / maxDistance;
  return Math.pow(normalized, 2); // Quadratic falloff for smoother influence
};

export const updatePhysics = (nodes: PhysicsNode[], deltaTime: number): PhysicsNode[] => {
  const dt = Math.min(deltaTime, 32) * 0.001; // Cap delta time
  const draggedNode = nodes.find(n => n.isDragging);
  
  return nodes.map(node => {
    if (node.isDragging || !node.originalX || !node.originalY) return node;

    let totalForceX = 0;
    let totalForceY = 0;

    // Return force
    const dx = node.originalX - node.x;
    const dy = node.originalY - node.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    // Progressive return force
    const returnStrength = Math.pow(distance / 100, 1.2) * config.RETURN_FORCE;
    totalForceX += dx * returnStrength;
    totalForceY += dy * returnStrength;

    // Influence from dragged node
    if (draggedNode) {
      const toDraggedX = draggedNode.x - node.x;
      const toDraggedY = draggedNode.y - node.y;
      const dragDistance = Math.sqrt(toDraggedX * toDraggedX + toDraggedY * toDraggedY);
      
      const influence = calculateInfluence(dragDistance, config.CONNECTION_RADIUS);
      totalForceX += toDraggedX * config.CONNECTION_STRENGTH * influence;
      totalForceY += toDraggedY * config.CONNECTION_STRENGTH * influence;
    }

    // Add wobble
    const time = Date.now() * 0.001;
    const wobbleX = Math.sin(time * config.WOBBLE_SPEED + node.x * 0.1) * config.WOBBLE_AMOUNT;
    const wobbleY = Math.cos(time * config.WOBBLE_SPEED + node.y * 0.1) * config.WOBBLE_AMOUNT;
    
    // Update velocity with smooth acceleration
    let newVelX = node.velocityX + (totalForceX + wobbleX) * dt;
    let newVelY = node.velocityY + (totalForceY + wobbleY) * dt;

    // Apply damping
    newVelX *= config.SPRING_DAMPING;
    newVelY *= config.SPRING_DAMPING;

    // Limit velocity
    const speed = Math.sqrt(newVelX * newVelX + newVelY * newVelY);
    if (speed > config.MAX_VELOCITY) {
      const scale = config.MAX_VELOCITY / speed;
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