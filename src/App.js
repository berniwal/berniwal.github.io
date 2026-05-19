// src/App.js
import React, { useEffect } from 'react';
import { HashRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { HashLink } from 'react-router-hash-link';
import './App.css';
import MainPage from './MainPage';
import Blog from './Blog';

// Reset window scroll on every route change so a new page doesn't inherit the
// previous route's scroll position (e.g. clicking a blog card from a scrolled
// blog list on the main page).
function ScrollToTopOnRoute() {
  const { pathname } = useLocation();
  useEffect(() => {
    // Skip if the user is navigating to a hash anchor on the same page —
    // HashLink already handles smooth-scroll in that case.
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
  }, [pathname]);
  return null;
}

function App() {
  return (
    <Router>
      <ScrollToTopOnRoute />
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
              <HashLink smooth to="/#projects">Experience</HashLink>
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
