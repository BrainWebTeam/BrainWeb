import React from 'react';
import { Hash } from 'lucide-react';

interface ToolsSidebarProps {
  currentTool: string;
}

function ToolsSidebar({ currentTool }: ToolsSidebarProps) {
  const tools = [
    {
      id: 'ml-tools',
      label: 'Machine Learning Tools',
      subItems: [
        { id: 'transformers', label: 'Transformer Tools' },
        { id: 'vector-db', label: 'Vector Database Tools' },
        { id: 'training', label: 'Training Tools' },
        { id: 'inference', label: 'Inference Tools' }
      ]
    },
    {
      id: 'data-tools',
      label: 'Data Tools',
      subItems: [
        { id: 'processing', label: 'Data Processing' },
        { id: 'validation', label: 'Data Validation' },
        { id: 'augmentation', label: 'Data Augmentation' }
      ]
    },
    {
      id: 'deployment-tools',
      label: 'Deployment Tools',
      subItems: [
        { id: 'model-serving', label: 'Model Serving' },
        { id: 'monitoring', label: 'Monitoring' },
        { id: 'scaling', label: 'Auto Scaling' }
      ]
    },
    {
      id: 'optimization-tools',
      label: 'Optimization Tools',
      subItems: [
        { id: 'quantization', label: 'Model Quantization' },
        { id: 'pruning', label: 'Model Pruning' },
        { id: 'distillation', label: 'Knowledge Distillation' }
      ]
    }
  ];

  return (
    <nav className="w-64 flex-shrink-0">
      <div className="sticky top-24 space-y-6">
        {tools.map(section => (
          <div key={section.id} className="space-y-1">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider px-4">
              {section.label}
            </h3>
            {section.subItems.map(item => (
              <a
                key={item.id}
                href={`#${item.id}`}
                className={`flex items-center gap-2 px-4 py-2 text-sm rounded-lg transition-colors ${
                  currentTool === item.id
                    ? 'text-white bg-gray-800'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                }`}
              >
                <Hash className="w-4 h-4" />
                {item.label}
              </a>
            ))}
          </div>
        ))}
      </div>
    </nav>
  );
}