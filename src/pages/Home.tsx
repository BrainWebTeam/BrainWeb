import React from 'react';
import NetworkVisual from '../components/NetworkVisual';

function Home() {
  return (
    <main className="relative pt-24 bg-[#050505]">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-center min-h-[80vh]">
          <NetworkVisual />
        </div>
      </div>
    </main>
  );
}

export default Home;