import React from 'react';
import { motion } from 'framer-motion';

interface ConnectionProps {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  strength: number;
  scale: number;
  sourceActivated: boolean;
  targetActivated: boolean;
  targetAvailable: boolean;
  delay?: number;
}

const Connection: React.FC<ConnectionProps> = ({ 
  x1, y1, x2, y2, 
  strength, 
  scale,
  sourceActivated,
  targetActivated,
  targetAvailable,
  delay = 0
}) => {
  // Only render if connection is active or available
  if (!sourceActivated && !targetActivated && !targetAvailable) {
    return null;
  }

  const dx = x2 - x1;
  const dy = y2 - y1;
  const distance = Math.sqrt(dx * dx + dy * dy);
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;
  const curvature = Math.min(distance * 0.2, 30);
  const pathData = `M ${x1} ${y1} Q ${midX} ${midY + curvature} ${x2} ${y2}`;

  const baseWidth = Math.max(0.5, Math.min(1.5, strength)) / scale;
  const activeWidth = baseWidth * 1.5;
  const pulseWidth = baseWidth * 2;

  return (
    <g className="connection">
      {/* Glow effect */}
      {(sourceActivated || targetActivated) && (
        <motion.path
          d={pathData}
          strokeWidth={activeWidth * 4}
          className="stroke-blue-500/10 blur-md"
          fill="none"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 0.6, delay: delay * 0.4 }}
        />
      )}

      {/* Active connection */}
      <motion.path
        d={pathData}
        strokeWidth={activeWidth}
        className={`
          ${targetAvailable 
            ? 'stroke-violet-500/30' 
            : sourceActivated && targetActivated
              ? 'stroke-blue-500'
              : 'stroke-blue-500/50'}
        `}
        fill="none"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ 
          pathLength: 1, 
          opacity: targetAvailable ? 0.3 : 0.8,
        }}
        transition={{ duration: 0.6, delay: delay * 0.4 }}
      />

      {/* Energy pulse effect */}
      {targetAvailable && (
        <motion.path
          d={pathData}
          strokeWidth={pulseWidth}
          className="stroke-violet-500/10"
          fill="none"
          animate={{
            pathOffset: [0, 1],
            opacity: [0.3, 0]
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "linear"
          }}
        />
      )}
    </g>
  );
};

export default Connection;