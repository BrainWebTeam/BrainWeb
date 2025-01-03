import React from 'react';
import { ZoomIn, ZoomOut } from 'lucide-react';

interface ZoomControlsProps {
  scale: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
}

const ZoomControls: React.FC<ZoomControlsProps> = ({ scale, onZoomIn, onZoomOut }) => {
  return (
    <div className="absolute bottom-4 right-4 flex flex-col gap-2 bg-white/80 backdrop-blur-sm rounded-lg p-2 shadow-lg">
      <button
        onClick={onZoomIn}
        className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        aria-label="Zoom in"
      >
        <ZoomIn className="w-5 h-5" />
      </button>
      <div className="text-center text-sm font-medium text-gray-600">
        {Math.round(scale * 100)}%
      </div>
      <button
        onClick={onZoomOut}
        className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        aria-label="Zoom out"
      >
        <ZoomOut className="w-5 h-5" />
      </button>
    </div>
  );
};

export default ZoomControls;