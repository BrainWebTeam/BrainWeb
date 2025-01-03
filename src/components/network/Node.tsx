import React from 'react';
import { motion } from 'framer-motion';

interface NodeProps {
  x: number;
  y: number;
  size?: number;
  activated?: boolean;
  available?: boolean;
  onClick?: () => void;
}

const Node: React.FC<NodeProps> = ({ 
  x, y, 
  size = 4, 
  activated = false,
  available = false,
  onClick
}) => {
  // Don't render inactive nodes
  if (!activated && !available) return null;

  return (
    <g>
      {/* Glow effect */}
      {(activated || available) && (
        <motion.circle
          cx={x}
          cy={y}
          r={size * 3}
          className={`
            ${activated ? 'fill-blue-500/20' : 'fill-violet-500/10'}
            blur-lg
          `}
          initial={false}
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      )}

      {/* Main node */}
      <motion.circle
        cx={x}
        cy={y}
        r={size}
        className={`
          cursor-pointer transition-colors
          ${activated 
            ? 'fill-blue-500 stroke-blue-300' 
            : 'fill-violet-500/80 stroke-violet-300/80'}
          stroke-[1.5]
        `}
        initial={false}
        animate={{
          scale: activated ? 1.2 : 1,
        }}
        onClick={onClick}
        whileHover={{
          scale: 1.3,
          transition: { duration: 0.2 }
        }}
      />

      {/* Core glow */}
      {activated && (
        <motion.circle
          cx={x}
          cy={y}
          r={size * 0.6}
          className="fill-blue-300"
          animate={{
            opacity: [0.6, 1, 0.6],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      )}
    </g>
  );
};

export default React.memo(Node);