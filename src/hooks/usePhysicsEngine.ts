import { useRef, useEffect, useCallback } from 'react';
import { PhysicsEngine } from '../utils/physics/engine';
import { PhysicsNode, Spring, Vector2D } from '../utils/physics/types';

export const usePhysicsEngine = (
  initialNodes: PhysicsNode[],
  springs: Spring[],
  bounds: { min: Vector2D; max: Vector2D }
) => {
  const engineRef = useRef<PhysicsEngine>();
  const rafRef = useRef<number>();
  const lastTimeRef = useRef<number>(0);

  // Initialize physics engine
  useEffect(() => {
    engineRef.current = new PhysicsEngine(bounds);
    
    // Add nodes and springs
    initialNodes.forEach(node => engineRef.current?.addNode(node));
    springs.forEach(spring => engineRef.current?.addSpring(spring));

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  const start = useCallback(() => {
    const animate = (timestamp: number) => {
      if (!engineRef.current) return;

      const deltaTime = lastTimeRef.current ? 
        Math.min(timestamp - lastTimeRef.current, 32) : 16.67;
      lastTimeRef.current = timestamp;

      engineRef.current.update(deltaTime);
      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);
  }, []);

  const stop = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = undefined;
    }
  }, []);

  const setNodePosition = useCallback((id: string, x: number, y: number) => {
    engineRef.current?.setNodePosition(id, { x, y });
  }, []);

  const startDrag = useCallback((id: string) => {
    engineRef.current?.startDrag(id);
  }, []);

  const endDrag = useCallback((id: string) => {
    engineRef.current?.endDrag(id);
  }, []);

  return {
    start,
    stop,
    setNodePosition,
    startDrag,
    endDrag,
    getState: () => engineRef.current?.getState(),
  };
};

export default usePhysicsEngine;