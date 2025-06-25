import React from 'react';
import './App.css';
import SignLanguageTranslator from './components/SignLanguageTranslator';

function App() {
  return (
    <div className="app">
      <div className="app-container">
        <SignLanguageTranslator />
      </div>
      <footer className="footer">
        <p>Built with ❤️ for HackPSU Spring 2025</p>
        <div className="tech-stack">
          <span>React</span>
          <span>Python</span>
          <span>MediaPipe</span>
          <span>Hugging Face</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
