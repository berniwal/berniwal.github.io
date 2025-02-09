// src/MainPage.js
import React from 'react';
import LandingPage from './LandingPage';
import ProjectPage from './ProjectPage';
import BlogPage from './BlogPage';
import ContactPage from './ContactPage';
import './MainPage.css';

const MainPage = () => {
  return (
    <div className="main-page">
      <section id="landing" className="section">
        <LandingPage />
      </section>
      <section id="projects" className="section">
        <ProjectPage />
      </section>
      <section id="blog" className="section">
        <BlogPage />
      </section>
      <section id="contact" className="section">
        <ContactPage />
      </section>
    </div>
  );
};

export default MainPage;
