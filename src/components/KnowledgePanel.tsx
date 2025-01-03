import React from 'react';
import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Node } from '../types/network';
import { NodeContent } from '../types/content';
import CodeBlock from './CodeBlock';

interface KnowledgePanelProps {
  node: Node | null;
  content: NodeContent | null;
  onClose: () => void;
}

const KnowledgePanel: React.FC<KnowledgePanelProps> = ({ node, content, onClose }) => {
  if (!node || !content) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: '100%', opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: '100%', opacity: 0 }}
        transition={{ type: "spring", stiffness: 100, damping: 20 }}
        className="fixed right-8 top-24 w-[400px] max-h-[80vh] bg-[#0A0A0A]/95 backdrop-blur-xl rounded-2xl overflow-hidden border border-gray-800/50 shadow-2xl shadow-black/50"
      >
        {/* Header */}
        <div className="relative">
          {/* Gradient line */}
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-blue-500/50 to-transparent" />
          
          <div className="px-6 py-4 backdrop-blur-sm bg-black/20">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
                  {content.title}
                </h2>
                <p className="text-xs text-blue-400/80 font-medium capitalize mt-0.5">
                  {node.type}
                </p>
              </div>
              <button
                onClick={onClose}
                className="p-1.5 text-gray-400 hover:text-white rounded-lg transition-colors hover:bg-white/5"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(80vh-80px)] custom-scrollbar">
          {/* Description */}
          <p className="text-gray-300 text-sm leading-relaxed">
            {content.description}
          </p>

          {/* Prerequisites */}
          {content.prerequisites && content.prerequisites.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-200 mb-2">Prerequisites</h3>
              <div className="flex flex-wrap gap-1.5">
                {content.prerequisites.map((prereq, index) => (
                  <span
                    key={index}
                    className="px-2 py-0.5 bg-blue-500/10 text-blue-400 rounded-full text-xs font-medium"
                  >
                    {prereq}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Key Concepts */}
          <div>
            <h3 className="text-sm font-semibold text-gray-200 mb-3">Key Concepts</h3>
            <ul className="space-y-2">
              {content.concepts.map((concept, index) => (
                <li key={index} className="flex items-start text-sm">
                  <span className="w-1.5 h-1.5 mt-1.5 mr-2 bg-blue-500 rounded-full flex-shrink-0" />
                  <span className="text-gray-300">{concept}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Code Examples */}
          {content.examples.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-200 mb-3">Examples</h3>
              <div className="space-y-4">
                {content.examples.map((example, index) => (
                  <div key={index}>
                    <p className="text-gray-400 text-xs mb-2">{example.description}</p>
                    <div className="bg-[#0D1117] rounded-xl overflow-hidden border border-gray-800/50">
                      <CodeBlock
                        code={example.code}
                        language={example.language}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Resources */}
          <div>
            <h3 className="text-sm font-semibold text-gray-200 mb-3">Learn More</h3>
            <div className="space-y-2">
              {content.resources.map((resource, index) => (
                <a
                  key={index}
                  href={resource.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block p-3 bg-white/5 hover:bg-white/10 rounded-xl transition-colors border border-gray-800/50"
                >
                  <h4 className="font-medium text-gray-200 text-sm mb-0.5">
                    {resource.title}
                  </h4>
                  <p className="text-gray-400 text-xs">
                    {resource.description}
                  </p>
                </a>
              ))}
            </div>
          </div>

          {/* Related Topics */}
          {content.relatedTopics && content.relatedTopics.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-200 mb-2">Related Topics</h3>
              <div className="flex flex-wrap gap-1.5">
                {content.relatedTopics.map((topic, index) => (
                  <span
                    key={index}
                    className="px-2 py-0.5 bg-white/5 text-gray-300 rounded-full text-xs"
                  >
                    {topic}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default KnowledgePanel;