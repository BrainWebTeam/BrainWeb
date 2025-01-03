import { Vector2D } from './types';

export const vector = {
  create: (x: number = 0, y: number = 0): Vector2D => ({ x, y }),
  
  add: (a: Vector2D, b: Vector2D): Vector2D => ({
    x: a.x + b.x,
    y: a.y + b.y,
  }),
  
  subtract: (a: Vector2D, b: Vector2D): Vector2D => ({
    x: a.x - b.x,
    y: a.y - b.y,
  }),
  
  multiply: (v: Vector2D, scalar: number): Vector2D => ({
    x: v.x * scalar,
    y: v.y * scalar,
  }),
  
  divide: (v: Vector2D, scalar: number): Vector2D => ({
    x: v.x / scalar,
    y: v.y / scalar,
  }),
  
  magnitude: (v: Vector2D): number => 
    Math.sqrt(v.x * v.x + v.y * v.y),
  
  normalize: (v: Vector2D): Vector2D => {
    const mag = vector.magnitude(v);
    return mag === 0 ? vector.create() : vector.divide(v, mag);
  },
  
  distance: (a: Vector2D, b: Vector2D): number => 
    vector.magnitude(vector.subtract(b, a)),
  
  lerp: (a: Vector2D, b: Vector2D, t: number): Vector2D => ({
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
  }),
  
  clamp: (v: Vector2D, min: number, max: number): Vector2D => {
    const mag = vector.magnitude(v);
    if (mag === 0) return v;
    const clamped = Math.max(min, Math.min(max, mag));
    return vector.multiply(vector.normalize(v), clamped);
  },
  
  angle: (v: Vector2D): number => 
    Math.atan2(v.y, v.x),
  
  rotate: (v: Vector2D, angle: number): Vector2D => ({
    x: v.x * Math.cos(angle) - v.y * Math.sin(angle),
    y: v.x * Math.sin(angle) + v.y * Math.cos(angle),
  }),
};