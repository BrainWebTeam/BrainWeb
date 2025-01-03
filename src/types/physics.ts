export interface Vector2D {
  x: number;
  y: number;
}

export interface PhysicsNode extends Node {
  originalX: number;
  originalY: number;
  velocityX: number;
  velocityY: number;
  isDragging: boolean;
}