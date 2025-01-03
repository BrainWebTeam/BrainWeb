import { useRef, useEffect } from 'react';

interface Vector {
  x: number;
  y: number;
}

interface PhysicsNode {
  id: string;
  position: Vector;
  velocity: Vector;
  force: Vector;
  mass: number;
  fixed: boolean;
}

interface Spring {
  nodeA: string;
  nodeB: string;
  length: number;
  stiffness: number;
}

const DAMPING = 0.8;
const SPRING_STIFFNESS = 0.08;
const REPULSION = 1000;

export const usePhysicsSimulation = (
  initialNodes: PhysicsNode[],
  connections: Spring[],
  onUpdate: (nodes: PhysicsNode[]) => void
) => {
  const nodesRef = useRef<PhysicsNode[]>(initialNodes);
  const rafRef = useRef<number>();

  const applyForce = (node: PhysicsNode, force: Vector) => {
    if (node.fixed) return;
    node.force.x += force.x;
    node.force.y += force.y;
  };

  const updatePhysics = () => {
    const nodes = nodesRef.current;
    
    // Reset forces
    nodes.forEach(node => {
      node.force = { x: 0, y: 0 };
    });

    // Apply spring forces
    connections.forEach(spring => {
      const nodeA = nodes.find(n => n.id === spring.nodeA)!;
      const nodeB = nodes.find(n => n.id === spring.nodeB)!;
      
      const dx = nodeB.position.x - nodeA.position.x;
      const dy = nodeB.position.y - nodeA.position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance === 0) return;
      
      const force = (distance - spring.length) * spring.stiffness;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;

      applyForce(nodeA, { x: fx, y: fy });
      applyForce(nodeB, { x: -fx, y: -fy });
    });

    // Apply repulsion forces between nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const nodeA = nodes[i];
        const nodeB = nodes[j];
        
        const dx = nodeB.position.x - nodeA.position.x;
        const dy = nodeB.position.y - nodeA.position.y;
        const distSq = dx * dx + dy * dy;
        
        if (distSq === 0) continue;
        
        const force = REPULSION / distSq;
        const fx = (dx / Math.sqrt(distSq)) * force;
        const fy = (dy / Math.sqrt(distSq)) * force;

        applyForce(nodeA, { x: -fx, y: -fy });
        applyForce(nodeB, { x: fx, y: fy });
      }
    }

    // Update velocities and positions
    nodes.forEach(node => {
      if (node.fixed) return;
      
      node.velocity.x = (node.velocity.x + node.force.x / node.mass) * DAMPING;
      node.velocity.y = (node.velocity.y + node.force.y / node.mass) * DAMPING;
      
      node.position.x += node.velocity.x;
      node.position.y += node.velocity.y;
    });

    onUpdate([...nodes]);
    rafRef.current = requestAnimationFrame(updatePhysics);
  };

  useEffect(() => {
    rafRef.current = requestAnimationFrame(updatePhysics);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  const dragNode = (id: string, position: Vector) => {
    const node = nodesRef.current.find(n => n.id === id);
    if (node) {
      node.position = position;
      node.velocity = { x: 0, y: 0 };
    }
  };

  return { dragNode };
};