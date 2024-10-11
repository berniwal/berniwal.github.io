import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import LandingPage from './LandingPage';
import ProjectPage from './ProjectPage';
import ContactPage from './ContactPage';

function App() {
  return (
    <Router>
      <div className="app-container">
        <nav className="navbar">
        <Link to="/" className="logo-link">
            <div className="logo">
              <img src={`${process.env.PUBLIC_URL}/favicon.ico`} alt="logo-icon" className="logo-icon" />
              <div className='logo-text'>Walser Tech</div>
            </div>
          </Link>
          <ul>
            <li><Link to="/">Home</Link></li>
            <li><Link to="/projects">Projects</Link></li>
            <li><Link to="/contact">Contact</Link></li>
          </ul>
        </nav>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/projects" element={<ProjectPage />} />
          <Route path="/contact" element={<ContactPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;