// src/ProjectPage.js
import React from 'react';
import './ProjectPage.css';
import { projects } from './projectData';

function ProjectPage() {
  return (
    <div className="project-page">
      <h2>Experience</h2>
      <div className="experience-container">
        {projects.map((project, index) => (
          <div className="row experience" key={index}>
            {/* Left Column: Timeline marker and date */}
            <div className="col-auto timeline-col d-none d-sm-flex">
              <div className="timeline-marker">
                <div className="timeline-date">{project.dateRange.split('–')[0].trim()}</div>
                <span className="timeline-dot"></span>
                {/* Optionally, you can show the lower end date or leave the bottom line as a connector */}
              </div>
            </div>
            {/* Right Column: Details card */}
            <div className="col experience-details py-2">
              <div className="card">
                <div className="card-body">
                  <div className="exp-title text-muted my-0">{project.title}</div>
                  <div className="exp-company text-muted my-0">
                    <a href={project.url} target="_blank" rel="noopener noreferrer">
                      {project.company}
                    </a>
                  </div>
                  <div className="exp-meta text-muted">
                    {project.dateRange}
                    <span className="middot-divider">•</span>
                    <span>{project.location}</span>
                  </div>
                  <div className="card-text">
                    {project.description}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ProjectPage;
