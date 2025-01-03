import React from 'react';

const Logo: React.FC = () => (
  <svg 
    width="36" 
    height="36" 
    viewBox="0 0 500 500" 
    fill="none" 
    xmlns="http://www.w3.org/2000/svg"
    className="relative z-10"
  >
    {/* Neural network nodes with glow effect */}
    <defs>
      <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
        <feMerge>
          <feMergeNode in="coloredBlur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
      <linearGradient id="nodeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#60A5FA" />
        <stop offset="100%" stopColor="#3B82F6" />
      </linearGradient>
    </defs>
    
    {/* Glowing nodes */}
    <circle cx="250" cy="250" r="47" fill="url(#nodeGradient)" filter="url(#glow)" />
    <circle cx="125" cy="125" r="31" fill="url(#nodeGradient)" filter="url(#glow)" />
    <circle cx="375" cy="125" r="31" fill="url(#nodeGradient)" filter="url(#glow)" />
    <circle cx="125" cy="375" r="31" fill="url(#nodeGradient)" filter="url(#glow)" />
    <circle cx="375" cy="375" r="31" fill="url(#nodeGradient)" filter="url(#glow)" />
    
    {/* Connection lines */}
    <path 
      d="M156 156L219 219M281 219L344 156M156 344L219 281M281 281L344 344" 
      className="stroke-blue-500" 
      strokeWidth="24" 
      strokeLinecap="round"
    />
    
    {/* Outer ring */}
    <circle 
      cx="250" 
      cy="250" 
      r="235" 
      className="stroke-blue-500" 
      strokeWidth="24" 
      strokeDasharray="63 63"
    />
  </svg>
);

export default Logo;