export interface Vector2D {
  x: number;
  y: number;
}

export const add = (a: Vector2D, b: Vector2D): Vector2D => ({
  x: a.x + b.x,
  y: a.y + b.y
});

export const subtract = (a: Vector2D, b: Vector2D): Vector2D => ({
  x: a.x - b.x,
  y: a.y - b.y
});

export const multiply = (v: Vector2D, scalar: number): Vector2D => ({
  x: v.x * scalar,
  y: v.y * scalar
});

export const magnitude = (v: Vector2D): number => 
  Math.sqrt(v.x * v.x + v.y * v.y);

export const normalize = (v: Vector2D): Vector2D => {
  const mag = magnitude(v);
  return mag === 0 ? { x: 0, y: 0 } : multiply(v, 1 / mag);
};