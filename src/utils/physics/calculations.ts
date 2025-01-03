import { Vector2D } from '../../types/physics';

export const calculateDistance = (p1: Vector2D, p2: Vector2D): number => {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  return Math.sqrt(dx * dx + dy * dy);
};

export const calculateInfluence = (distance: number, maxDistance: number): number => {
  if (distance >= maxDistance) return 0;
  const normalized = 1 - distance / maxDistance;
  return Math.pow(normalized, 2);
};