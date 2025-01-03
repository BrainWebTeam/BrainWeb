import { useState, useCallback, useRef } from 'react';

interface Position {
  x: number;
  y: number;
}

export const usePan = () => {
  const [position, setPosition] = useState<Position>({ x: 0, y: 0 });
  const isPanning = useRef(false);
  const startPan = useRef<Position>({ x: 0, y: 0 });
  const lastValidPosition = useRef<Position>({ x: 0, y: 0 });

  const onPanStart = useCallback((e: React.MouseEvent) => {
    isPanning.current = true;
    startPan.current = {
      x: e.clientX - position.x,
      y: e.clientY - position.y
    };
    lastValidPosition.current = position;
  }, [position]);

  const onPanMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning.current) return;

    const newPosition = {
      x: e.clientX - startPan.current.x,
      y: e.clientY - startPan.current.y
    };

    // Limit the panning range to prevent extreme movements
    const maxPan = 1000;
    newPosition.x = Math.max(-maxPan, Math.min(maxPan, newPosition.x));
    newPosition.y = Math.max(-maxPan, Math.min(maxPan, newPosition.y));

    lastValidPosition.current = newPosition;
    setPosition(newPosition);
  }, []);

  const onPanEnd = useCallback(() => {
    isPanning.current = false;
  }, []);

  return {
    position,
    handlers: {
      onMouseDown: onPanStart,
      onMouseMove: onPanMove,
      onMouseUp: onPanEnd,
      onMouseLeave: onPanEnd
    }
  };
};