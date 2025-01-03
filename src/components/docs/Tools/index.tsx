import React, { lazy, Suspense, useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';

// Lazy load tool components
const AmadeusTools = lazy(() => import('./AmadeusTools'));
const AudioTools = lazy(() => import('./AudioTools'));
const CalculatorTools = lazy(() => import('./CalculatorTools'));
const ConversationTools = lazy(() => import('./ConversationTools'));
const EmbeddingsTools = lazy(() => import('./EmbeddingsTools'));
const FileTools = lazy(() => import('./FileTools'));
const GitHubTools = lazy(() => import('./GitHubTools'));
const LangChainTools = lazy(() => import('./LangChainTools'));
const PineconeTools = lazy(() => import('./PineconeTools'));
const TextSplitters = lazy(() => import('./TextSplitters'));
const WebTools = lazy(() => import('./WebTools'));
const WikipediaTools = lazy(() => import('./WikipediaTools'));
const YahooFinanceTools = lazy(() => import('./YahooFinanceTools'));

// Loading component
const ToolSkeleton = () => (
  <div className="animate-pulse">
    <div className="h-8 w-48 bg-gray-700 rounded mb-4"></div>
    <div className="space-y-3">
      <div className="h-4 w-full bg-gray-700 rounded"></div>
      <div className="h-4 w-5/6 bg-gray-700 rounded"></div>
      <div className="h-4 w-4/6 bg-gray-700 rounded"></div>
    </div>
  </div>
);

// Tool components map
const TOOL_COMPONENTS = {
  'amadeus-tools': AmadeusTools,
  'audio-tools': AudioTools,
  'calculator-tools': CalculatorTools,
  'conversation-tools': ConversationTools,
  'embeddings-tools': EmbeddingsTools,
  'file-tools': FileTools,
  'github-tools': GitHubTools,
  'langchain-tools': LangChainTools,
  'pinecone-tools': PineconeTools,
  'text-splitters': TextSplitters,
  'web-tools': WebTools,
  'wikipedia-tools': WikipediaTools,
  'yahoo-finance-tools': YahooFinanceTools
};

function Tools() {
  const location = useLocation();
  const [activeToolId, setActiveToolId] = useState<string | null>(null);

  useEffect(() => {
    // Extract tool ID from hash
    const hash = location.hash.slice(1);
    if (hash && hash !== activeToolId) {
      setActiveToolId(hash);
      // Smooth scroll to tool section
      const element = document.getElementById(hash);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  }, [location.hash]);

  // Render only the active tool
  const renderTool = () => {
    if (!activeToolId) return null;

    const ToolComponent = TOOL_COMPONENTS[activeToolId as keyof typeof TOOL_COMPONENTS];
    if (!ToolComponent) return null;

    return (
      <Suspense fallback={<ToolSkeleton />}>
        <ToolComponent />
      </Suspense>
    );
  };

  return (
    <div className="prose prose-invert max-w-none">
      <h1>Tools</h1>
      <p className="lead">
        Advanced tools and utilities for machine learning, data processing, optimization,
        and system management. Each tool provides enterprise-grade functionality with
        comprehensive configuration options and robust error handling.
      </p>

      <div className="space-y-16">
        {renderTool()}
      </div>
    </div>
  );
}

export default Tools;