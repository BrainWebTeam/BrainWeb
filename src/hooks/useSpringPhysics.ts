import { useRef, useEffect, useCallback } from 'react';
import { PhysicsNode } from '../types/physics';
import { applySpringForces } from '../utils/physics';
import { applyReturnForces } from '../utils/forces';
import { PHYSICS_CONFIG } from '../config/physics';
import { Vector2D } from '../types/physics';

interface Spring {
  nodeA: string;
  nodeB: string;
  length: number;
  stiffness: number;
}

export const useSpringPhysics = (
  initialNodes: PhysicsNode[],
  connections: Spring[],
  onUpdate: (nodes: PhysicsNode[]) => void
) => {
  const nodesRef = useRef<PhysicsNode[]>(initialNodes);
  const rafRef = useRef<number>();
  const lastTimeRef = useRef<number>(0);

  const updatePhysics = useCallback((timestamp: number) => {
    const deltaTime = Math.min(
      lastTimeRef.current ? timestamp - lastTimeRef.current : 16.67,
      32 // Cap max delta time to prevent instability
    );
    lastTimeRef.current = timestamp;

    const nodes = nodesRef.current.map(node => ({ ...node }));

    // Reset forces
    nodes.forEach(node => {
      node.force = { x: 0, y: 0 };
    });

    // Apply forces
    applySpringForces(nodes, connections, PHYSICS_CONFIG);
    applyReturnForces(nodes, PHYSICS_CONFIG);

    // Update positions and velocities
    let isMoving = false;
    nodes.forEach(node => {
      if (!node.isDragging) {
        // Update velocity with force and deltaTime scaling
        const dt = deltaTime * 0.001;
        node.velocity.x += node.force.x * dt;
        node.velocity.y += node.force.y * dt;

        // Apply damping
        node.velocity.x *= PHYSICS_CONFIG.SPRING_DAMPING;
        node.velocity.y *= PHYSICS_CONFIG.SPRING_DAMPING;

        // Clamp velocity
        const speed = Math.sqrt(
          node.velocity.x * node.velocity.x + 
          node.velocity.y * node.velocity.y
        );
        if (speed > PHYSICS_CONFIG.MAX_VELOCITY) {
          const scale = PHYSICS_CONFIG.MAX_VELOCITY / speed;
          node.velocity.x *= scale;
          node.velocity.y *= scale;
        }

        // Update position
        node.position.x += node.velocity.x * dt;
        node.position.y += node.velocity.y * dt;

        // Check if still moving
        if (Math.abs(node.velocity.x) > 0.01 || Math.abs(node.velocity.y) > 0.01) {
          isMoving = true;
        }
      }
    });

    nodesRef.current = nodes;
    onUpdate(nodes);

    if (isMoving) {
      rafRef.current = requestAnimationFrame(updatePhysics);
    }
  }, [connections, onUpdate]);

  const startDrag = useCallback((id: string) => {
    const nodes = nodesRef.current.map(node => ({
      ...node,
      isDragging: node.id === id ? true : node.isDragging,
      velocity: node.id === id ? { x: 0, y: 0 } : node.velocity
    }));
    nodesRef.current = nodes;
    onUpdate(nodes);

    // Start physics simulation if not running
    if (!rafRef.current) {
      rafRef.current = requestAnimationFrame(updatePhysics);
    }
  }, [updatePhysics, onUpdate]);

  const endDrag = useCallback((id: string) => {
    const nodes = nodesRef.current.map(node => ({
      ...node,
      isDragging: node.id === id ? false : node.isDragging,
      justReleased: node.id === id ? true : node.justReleased
    }));
    nodesRef.current = nodes;
    onUpdate(nodes);

    // Ensure physics simulation is running
    if (!rafRef.current) {
      rafRef.current = requestAnimationFrame(updatePhysics);
    }
  }, [updatePhysics, onUpdate]);

  const updateNodePosition = useCallback((id: string, position: Vector2D) => {
    const nodes = nodesRef.current.map(node => 
      node.id === id ? { ...node, position } : node
    );
    nodesRef.current = nodes;
    onUpdate(nodes);
  }, [onUpdate]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(updatePhysics);
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [updatePhysics]);

  return { startDrag, endDrag, updateNodePosition };
};

export default useSpringPhysics;