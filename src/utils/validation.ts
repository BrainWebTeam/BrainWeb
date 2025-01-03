import { Vector2D } from '../types/physics';

export const isValidPosition = (position: Vector2D): boolean => {
  return (
    position &&
    typeof position.x === 'number' &&
    typeof position.y === 'number' &&
    !isNaN(position.x) &&
    !isNaN(position.y) &&
    isFinite(position.x) &&
    isFinite(position.y)
  );
};