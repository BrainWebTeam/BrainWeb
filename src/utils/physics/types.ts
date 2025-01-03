export interface Vector2D {
  x: number;
  y: number;
}

export interface PhysicsNode {
  id: string;
  position: Vector2D;
  velocity: Vector2D;
  force: Vector2D;
  mass: number;
  damping: number;
  fixed: boolean;
  originalPosition: Vector2D;
  connections: string[];
  isDragging: boolean;
  type: 'center' | 'primary' | 'secondary';
}

export interface Spring {
  nodeA: string;
  nodeB: string;
  restLength: number;
  stiffness: number;
}

export interface PhysicsState {
  nodes: Map<string, PhysicsNode>;
  springs: Spring[];
  time: number;
  deltaTime: number;
}