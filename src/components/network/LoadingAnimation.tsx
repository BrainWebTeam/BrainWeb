import React from 'react';
import { motion } from 'framer-motion';

const LoadingAnimation: React.FC = () => {
  return (
    <g>
      {/* Central pulse */}
      <motion.circle
        cx="400"
        cy="400"
        r="40"
        className="fill-blue-500/20 blur-md"
        initial={{ scale: 0, opacity: 0 }}
        animate={{
          scale: [1, 2, 1],
          opacity: [0.6, 0.2, 0.6],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />

      {/* Orbiting particles */}
      {[...Array(8)].map((_, i) => {
        const angle = (i * Math.PI * 2) / 8;
        const delay = i * 0.2;
        return (
          <motion.circle
            key={i}
            cx="400"
            cy="400"
            r="2"
            className="fill-blue-400"
            initial={{ 
              opacity: 0,
              scale: 0,
            }}
            animate={{
              opacity: [0, 1, 0],
              scale: [0, 1, 0],
              x: [0, Math.cos(angle) * 60, 0],
              y: [0, Math.sin(angle) * 60, 0],
            }}
            transition={{
              duration: 3,
              delay,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        );
      })}

      {/* Core */}
      <motion.circle
        cx="400"
        cy="400"
        r="8"
        className="fill-blue-500 stroke-blue-300 stroke-2"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{
          type: "spring",
          stiffness: 200,
          damping: 10
        }}
      />
    </g>
  );
};

export default LoadingAnimation;