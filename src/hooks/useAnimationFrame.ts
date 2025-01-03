import { useEffect, useRef } from 'react';

export const useAnimationFrame = (callback: (deltaTime: number) => void) => {
  const requestRef = useRef<number>();
  const previousTimeRef = useRef<number>();
  const isActiveRef = useRef(true);

  useEffect(() => {
    const animate = (time: number) => {
      if (!isActiveRef.current) return;

      if (previousTimeRef.current !== undefined) {
        const deltaTime = Math.min(time - previousTimeRef.current, 32); // Cap at ~30fps
        callback(deltaTime);
      }
      previousTimeRef.current = time;
      requestRef.current = requestAnimationFrame(animate);
    };

    requestRef.current = requestAnimationFrame(animate);
    
    return () => {
      isActiveRef.current = false;
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [callback]);
};