import React from 'react';
import { Link } from 'react-router-dom';
import Logo from './Logo';

const Header: React.FC = () => {
  const [showLaunchSoon, setShowLaunchSoon] = React.useState(false);

  return (
    <header className="fixed top-0 w-full z-50">
      {/* Gradient border */}
      <div className="h-[1px] w-full bg-gradient-to-r from-transparent via-blue-500/50 to-transparent" />
      
      {/* Header content */}
      <div className="bg-[#0A0A0A]/80 backdrop-blur-xl">
        <div className="container mx-auto px-6 py-4">
          <nav className="flex items-center justify-between">
            {/* Logo */}
            <Link 
              to="/" 
              className="flex items-center space-x-3 group"
            >
              <div className="relative">
                <Logo />
                <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-xl group-hover:bg-blue-400/30 transition-colors" />
              </div>
              <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
                CogniNet
              </span>
            </Link>

            {/* Navigation */}
            <div className="flex items-center gap-3">
              {/* Documentation */}
              <Link 
                to="/docs" 
                className="px-4 py-2 text-gray-300 hover:text-white transition-colors"
              >
                Documentation
              </Link>

              {/* GitHub */}
              <a
                href="https://github.com/cogninet"
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 text-gray-300 hover:text-white transition-colors"
              >
                GitHub
              </a>

              {/* Divider */}
              <div className="w-px h-6 bg-gray-800" />

              {/* Social Link */}
              <a
                href="https://x.com/CogniNetowork"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center w-9 h-9 rounded-lg bg-white/5 hover:bg-white/10 text-gray-300 transition-all"
              >
                <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                </svg>
              </a>

              {/* Buy Button */}
              <div className="relative">
                <button
                  onClick={() => setShowLaunchSoon(true)}
                  onMouseLeave={() => setShowLaunchSoon(false)}
                  className="px-5 py-2 rounded-lg bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 font-medium transition-all"
                >
                  Buy Now
                </button>
                {showLaunchSoon && (
                  <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 px-3 py-1.5 bg-[#0A0A0A] text-gray-300 text-sm rounded-lg whitespace-nowrap animate-fade-in border border-gray-800/50">
                    Launching soon
                  </div>
                )}
              </div>
            </div>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;