import React, { useState } from 'react';
import { Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { ChevronDown, ChevronRight, Search, Book, Wrench, Zap, Lock, Code, Activity } from 'lucide-react';
import Introduction from '../components/docs/Introduction';
import CoreConcepts from '../components/docs/CoreConcepts';
import GettingStarted from '../components/docs/GettingStarted';
import NetworkArchitecture from '../components/docs/NetworkArchitecture';
import InteractionSystem from '../components/docs/InteractionSystem';
import LearningSystem from '../components/docs/LearningSystem';
import AdvancedFeatures from '../components/docs/AdvancedFeatures';
import DataManagement from '../components/docs/DataManagement';
import UserExperience from '../components/docs/UserExperience';
import Performance from '../components/docs/Performance';
import Security from '../components/docs/Security';
import APIIntegration from '../components/docs/APIIntegration';
import Tools from '../components/docs/Tools';

function Docs() {
  const [toolsExpanded, setToolsExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const location = useLocation();

  const sections = [
    { path: '/docs', icon: Book, label: 'Introduction' },
    { path: '/docs/core-concepts', icon: Zap, label: 'Core Concepts' },
    { path: '/docs/getting-started', icon: Activity, label: 'Getting Started' },
    { path: '/docs/network-architecture', icon: Code, label: 'Network Architecture' },
    { path: '/docs/interaction-system', icon: Wrench, label: 'Interaction System' },
    { path: '/docs/learning-system', icon: Book, label: 'Learning System' },
    { path: '/docs/advanced-features', icon: Zap, label: 'Advanced Features' },
    { path: '/docs/data-management', icon: Code, label: 'Data Management' },
    { path: '/docs/user-experience', icon: Wrench, label: 'User Experience' },
    { path: '/docs/performance', icon: Activity, label: 'Performance' },
    { path: '/docs/security', icon: Lock, label: 'Security' },
    { path: '/docs/api-integration', icon: Code, label: 'API Integration' }
  ];

  const tools = [
    { id: 'amadeus-tools', label: 'Amadeus Tools' },
    { id: 'audio-tools', label: 'Audio Tools' },
    { id: 'calculator-tools', label: 'Calculator Tools' },
    { id: 'conversation-tools', label: 'Conversation Tools' },
    { id: 'embeddings-tools', label: 'Embeddings Tools' },
    { id: 'file-tools', label: 'File Tools' },
    { id: 'github-tools', label: 'GitHub Tools' },
    { id: 'langchain-tools', label: 'LangChain Tools' },
    { id: 'pinecone-tools', label: 'Pinecone Tools' },
    { id: 'text-splitters', label: 'Text Splitters' },
    { id: 'web-tools', label: 'Web Tools' },
    { id: 'wikipedia-tools', label: 'Wikipedia Tools' },
    { id: 'yahoo-finance-tools', label: 'Yahoo Finance Tools' }
  ];

  const filteredTools = tools.filter(tool =>
    tool.label.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const currentToolId = location.hash.slice(1);

  return (
    <div className="min-h-screen bg-[#050505]">
      <div className="container mx-auto px-6 pt-24">
        <div className="flex gap-12">
          {/* Sidebar Navigation */}
          <nav className="w-72 flex-shrink-0">
            <div className="sticky top-24 space-y-6">
              {/* Search Bar */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-blue-400" />
                <input
                  type="text"
                  placeholder="Search documentation..."
                  className="w-full pl-10 pr-4 py-2.5 bg-[#0A0A0A] border border-gray-800/50 rounded-xl text-gray-300 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>

              {/* Navigation Links */}
              <div className="space-y-1">
                {sections.map(({ path, icon: Icon, label }) => (
                  <NavLink
                    key={path}
                    to={path}
                    end={path === '/docs'}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all ${
                        isActive
                          ? 'bg-blue-500/10 text-blue-400 shadow-lg shadow-blue-500/5'
                          : 'text-gray-400 hover:text-blue-400 hover:bg-white/5'
                      }`
                    }
                  >
                    <Icon className="w-4 h-4" />
                    <span className="font-medium">{label}</span>
                  </NavLink>
                ))}

                {/* Tools Section */}
                <div>
                  <button
                    onClick={() => setToolsExpanded(!toolsExpanded)}
                    className="w-full flex items-center justify-between px-4 py-2.5 text-gray-400 hover:text-blue-400 hover:bg-white/5 rounded-xl transition-all"
                  >
                    <div className="flex items-center gap-3">
                      <Wrench className="w-4 h-4" />
                      <span className="font-medium">Tools</span>
                    </div>
                    {toolsExpanded ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                  </button>

                  {toolsExpanded && (
                    <div className="ml-4 space-y-1 mt-1">
                      {filteredTools.map(tool => (
                        <NavLink
                          key={tool.id}
                          to={`/docs/tools#${tool.id}`}
                          className={`flex items-center gap-2 px-4 py-2 text-sm rounded-xl transition-all ${
                            currentToolId === tool.id
                              ? 'bg-blue-500/10 text-blue-400 shadow-lg shadow-blue-500/5'
                              : 'text-gray-400 hover:text-blue-400 hover:bg-white/5'
                          }`}
                          onClick={() => {
                            if (location.hash) {
                              window.history.pushState(
                                "", 
                                document.title, 
                                window.location.pathname + window.location.search
                              );
                            }
                            setTimeout(() => {
                              window.location.hash = tool.id;
                            }, 0);
                          }}
                        >
                          {tool.label}
                        </NavLink>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </nav>

          {/* Main Content */}
          <div className="flex-1 min-w-0 pb-16">
            <div className="prose prose-invert prose-blue max-w-none">
              <Routes>
                <Route path="/" element={<Introduction />} />
                <Route path="/core-concepts" element={<CoreConcepts />} />
                <Route path="/tools" element={<Tools />} />
                <Route path="/getting-started" element={<GettingStarted />} />
                <Route path="/network-architecture" element={<NetworkArchitecture />} />
                <Route path="/interaction-system" element={<InteractionSystem />} />
                <Route path="/learning-system" element={<LearningSystem />} />
                <Route path="/advanced-features" element={<AdvancedFeatures />} />
                <Route path="/data-management" element={<DataManagement />} />
                <Route path="/user-experience" element={<UserExperience />} />
                <Route path="/performance" element={<Performance />} />
                <Route path="/security" element={<Security />} />
                <Route path="/api-integration" element={<APIIntegration />} />
              </Routes>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Docs;