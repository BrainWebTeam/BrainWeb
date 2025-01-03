import { PhysicsNode, Spring, PhysicsState, Vector2D } from './types';
import { vector } from './vector';
import { forces } from './forces';
import { integration } from './integration';
import { constraints } from './constraints';
import { PHYSICS } from './constants';

export class PhysicsEngine {
  private state: PhysicsState;
  private bounds: { min: Vector2D; max: Vector2D };

  constructor(bounds: { min: Vector2D; max: Vector2D }) {
    this.bounds = bounds;
    this.state = {
      nodes: new Map(),
      springs: [],
      time: 0,
      deltaTime: 0
    };
  }

  addNode(node: PhysicsNode): void {
    this.state.nodes.set(node.id, { ...node });
  }

  addSpring(spring: Spring): void {
    this.state.springs.push({ ...spring });
  }

  private applyForces(): void {
    for (const node of this.state.nodes.values()) {
      if (node.isDragging) continue;

      // Apply spring forces
      for (const spring of this.state.springs) {
        if (spring.nodeA === node.id || spring.nodeB === node.id) {
          const otherNode = this.state.nodes.get(
            spring.nodeA === node.id ? spring.nodeB : spring.nodeA
          );
          if (otherNode) {
            const springForce = forces.spring(node, otherNode, spring.restLength, spring.stiffness);
            node.force = vector.add(node.force, springForce);
          }
        }
      }

      // Apply other forces
      node.force = vector.add(node.force, forces.drag(node));
      node.force = vector.add(node.force, forces.return(node));

      // Apply neighbor influence
      const neighbors = node.connections
        .map(id => this.state.nodes.get(id))
        .filter((n): n is PhysicsNode => n !== undefined);
      node.force = vector.add(node.force, forces.neighborInfluence(node, neighbors));
    }
  }

  update(deltaTime: number): void {
    const dt = Math.min(deltaTime / PHYSICS.ANIMATION.SUBSTEPS, PHYSICS.ANIMATION.MAX_DELTA);
    
    for (let i = 0; i < PHYSICS.ANIMATION.SUBSTEPS; i++) {
      this.updateSubstep(dt);
    }
    
    this.state.time += deltaTime;
    this.state.deltaTime = deltaTime;
  }

  private updateSubstep(dt: number): void {
    // Reset forces
    for (const node of this.state.nodes.values()) {
      node.force = vector.create();
    }
    
    // Apply forces
    this.applyForces();
    
    // Update positions and velocities
    for (const node of this.state.nodes.values()) {
      if (!node.isDragging) {
        integration.verlet(node, dt);
        constraints.boundingBox(node, this.bounds);
      }
    }

    // Apply spring constraints multiple times for stability
    for (let i = 0; i < 2; i++) {
      for (const spring of this.state.springs) {
        const nodeA = this.state.nodes.get(spring.nodeA);
        const nodeB = this.state.nodes.get(spring.nodeB);
        if (nodeA && nodeB) {
          constraints.distance(nodeA, nodeB, spring.restLength * 0.9, spring.restLength * 1.1);
        }
      }
    }
  }

  startDrag(id: string): void {
    const node = this.state.nodes.get(id);
    if (node) {
      node.isDragging = true;
      node.velocity = vector.create();
    }
  }

  endDrag(id: string): void {
    const node = this.state.nodes.get(id);
    if (node) {
      node.isDragging = false;
    }
  }

  setNodePosition(id: string, position: Vector2D): void {
    const node = this.state.nodes.get(id);
    if (node) {
      node.position = { ...position };
    }
  }

  getState(): PhysicsState {
    return this.state;
  }
}

export default PhysicsEngine;