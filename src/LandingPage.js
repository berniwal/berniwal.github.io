// src/LandingPage.js
import React from 'react';
import { FaLinkedin, FaGithub, FaEnvelope } from 'react-icons/fa';
import { HashLink } from 'react-router-hash-link';
import './LandingPage.css';

const LandingPage = () => {
  return (
    <div className="landing-page">
      <div className="intro">
        <h1>Bernhard Walser</h1>
        <p>Machine Learning Engineer</p>
      </div>
      <div className="social-links">
        <a
          href="https://www.linkedin.com/in/bernhardwalser/"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="LinkedIn"
        >
          <FaLinkedin className="social-icon" />
        </a>
        <a
          href="https://github.com/berniwal"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="GitHub"
        >
          <FaGithub className="social-icon" />
        </a>
        <HashLink smooth to="/#contact"><FaEnvelope className="social-icon" /></HashLink>
      </div>
    </div>
  );
};

export default LandingPage;
