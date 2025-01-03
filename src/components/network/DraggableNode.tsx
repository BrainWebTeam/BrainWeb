import React from 'react';
import { motion } from 'framer-motion';

interface DraggableNodeProps {
  id: string;
  x: number;
  y: number;
  type: 'center' | 'primary' | 'secondary';
  isDragging?: boolean;
  onDragStart: () => void;
  onDrag: (id: string, x: number, y: number) => void;
  onDragEnd: () => void;
}

const DraggableNode: React.FC<DraggableNodeProps> = ({
  id,
  x,
  y,
  type,
  isDragging,
  onDragStart,
  onDrag,
  onDragEnd
}) => {
  const size = type === 'center' ? 12 : type === 'primary' ? 6 : 3;
  const isPulsing = type === 'center';

  // Ensure coordinates are valid numbers
  const safeX = Number.isFinite(x) ? x : 0;
  const safeY = Number.isFinite(y) ? y : 0;

  const handleDrag = (event: any, info: { point: { x: number; y: number } }) => {
    const svg = event.target.closest('svg');
    if (!svg) return;

    const point = svg.createSVGPoint();
    point.x = info.point.x;
    point.y = info.point.y;
    
    const svgPoint = point.matrixTransform(svg.getScreenCTM()?.inverse());
    onDrag(id, svgPoint.x, svgPoint.y);
  };

  return (
    <g>
      {isPulsing && !isDragging && (
        <>
          <circle 
            cx={safeX} 
            cy={safeY} 
            r={size * 2} 
            className="fill-black/5"
          />
          <circle 
            cx={safeX} 
            cy={safeY} 
            r={size * 1.5} 
            className="fill-black/5 animate-pulse"
          />
        </>
      )}
      <motion.circle 
        cx={safeX}
        cy={safeY}
        r={size}
        className={`
          cursor-grab active:cursor-grabbing transition-all duration-150
          ${type === 'center' ? 'fill-black' : 
            type === 'primary' ? 'fill-white stroke-black stroke-[1.5]' : 
            'fill-white stroke-gray-300 stroke-[1]'}
        `}
        drag
        dragMomentum={false}
        onDragStart={onDragStart}
        onDrag={handleDrag}
        onDragEnd={onDragEnd}
        dragConstraints={{ left: 0, right: 800, top: 0, bottom: 800 }}
        dragElastic={0.1}
      />
    </g>
  );
};

export default DraggableNode;