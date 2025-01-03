import { useState, useCallback } from 'react';

interface Position {
  x: number;
  y: number;
}

const MIN_SCALE = 0.5;
const MAX_SCALE = 2;
const ZOOM_STEP = 0.1;

export const useZoom = () => {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState<Position>({ x: 0, y: 0 });

  const onZoomIn = useCallback(() => {
    setScale(prev => Math.min(prev + ZOOM_STEP, MAX_SCALE));
  }, []);

  const onZoomOut = useCallback(() => {
    setScale(prev => Math.max(prev - ZOOM_STEP, MIN_SCALE));
  }, []);

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    
    // Update scale
    const delta = -Math.sign(e.deltaY) * ZOOM_STEP;
    const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale + delta));
    
    if (newScale !== scale) {
      // Calculate new position to zoom towards cursor
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const scaleRatio = 1 - newScale / scale;
      setPosition(prev => ({
        x: prev.x + (x - prev.x) * scaleRatio,
        y: prev.y + (y - prev.y) * scaleRatio
      }));
      setScale(newScale);
    }
  }, [scale]);

  return {
    scale,
    position,
    onZoomIn,
    onZoomOut,
    onWheel
  };
};