import React from 'react';
import { Link } from 'react-router-dom';
import './LandingPage.css';


function LandingPage() {
  return (
    <div className="landing-page">
      <h1>Welcome to Walser Tech</h1>
      <p>We develop innovative web-based applications and AI tools to solve modern problems.</p>
      <Link to="/projects">
        <button className="cta-button">Our projects.</button>
      </Link>
    </div>
  );
}

export default LandingPage;