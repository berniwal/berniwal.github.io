// src/App.js
import React from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import { HashLink } from 'react-router-hash-link';
import './App.css';
import MainPage from './MainPage';
import Blog from './Blog';

function App() {
  return (
    <Router>
      <div className="app-container">
        <nav className="navbar">
          <HashLink smooth to="/#landing" className="logo-link">
            <div className="logo">
              <img
                src={`${process.env.PUBLIC_URL}/favicon.ico`}
                alt="logo-icon"
                className="logo-icon"
              />
              <div className="logo-text">Bernhard Walser</div>
            </div>
          </HashLink>
          <ul>
            <li>
              <HashLink smooth to="/#blog">Blog</HashLink>
            </li>
            <li>
              <HashLink smooth to="/#projects">Projects</HashLink>
            </li>
            <li>
              <HashLink smooth to="/#contact">Contact</HashLink>
            </li>
          </ul>
        </nav>
        <Routes>
          <Route path="/" element={<MainPage />} />
          {/* Keep this route separate for individual blog posts */}
          <Route path="/blog/:slug" element={<Blog />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
