import { PhysicsNode, Vector2D } from './types';
import { PHYSICS } from './constants';
import { vector } from './vector';

export const forces = {
  spring: (nodeA: PhysicsNode, nodeB: PhysicsNode, restLength: number, stiffness: number): Vector2D => {
    const displacement = vector.subtract(nodeB.position, nodeA.position);
    const distance = vector.magnitude(displacement);
    
    if (distance === 0) return vector.create();
    
    const stretch = distance - restLength;
    const force = stretch * stiffness;
    
    return vector.multiply(vector.normalize(displacement), force);
  },
  
  attraction: (nodeA: PhysicsNode, nodeB: PhysicsNode): Vector2D => {
    const displacement = vector.subtract(nodeB.position, nodeA.position);
    const distance = vector.magnitude(displacement);
    
    if (distance === 0 || distance > PHYSICS.DISTANCE.INFLUENCE_MAX) {
      return vector.create();
    }
    
    const force = (PHYSICS.FORCE.ATTRACTION * nodeA.mass * nodeB.mass) / (distance * distance);
    return vector.multiply(vector.normalize(displacement), force);
  },
  
  repulsion: (nodeA: PhysicsNode, nodeB: PhysicsNode): Vector2D => {
    const displacement = vector.subtract(nodeB.position, nodeA.position);
    const distance = vector.magnitude(displacement);
    
    if (distance === 0 || distance > PHYSICS.DISTANCE.INFLUENCE_MAX) {
      return vector.create();
    }
    
    const force = -(PHYSICS.FORCE.REPULSION * nodeA.mass * nodeB.mass) / (distance * distance);
    return vector.multiply(vector.normalize(displacement), force);
  },
  
  drag: (node: PhysicsNode): Vector2D => 
    vector.multiply(node.velocity, -PHYSICS.FORCE.DRAG),
  
  return: (node: PhysicsNode): Vector2D => {
    const displacement = vector.subtract(node.originalPosition, node.position);
    const distance = vector.magnitude(displacement);
    
    if (distance < PHYSICS.DISTANCE.INFLUENCE_MIN) return vector.create();
    
    const force = distance * PHYSICS.FORCE.RETURN;
    return vector.multiply(vector.normalize(displacement), force);
  },
  
  neighborInfluence: (node: PhysicsNode, neighbors: PhysicsNode[]): Vector2D => {
    if (neighbors.length === 0) return vector.create();
    
    const averagePosition = neighbors.reduce(
      (sum, neighbor) => vector.add(sum, neighbor.position),
      vector.create()
    );
    
    vector.divide(averagePosition, neighbors.length);
    const displacement = vector.subtract(averagePosition, node.position);
    return vector.multiply(displacement, PHYSICS.INTERACTION.NEIGHBOR_INFLUENCE);
  },
};