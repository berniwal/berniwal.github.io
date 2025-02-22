// src/BlogPage.js
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './BlogPage.css';
import { blogPosts } from './blogPosts';

function BlogPage() {
  // Track the chosen category
  const [chosenCategory, setChosenCategory] = useState('All');
  // Track the current page for pagination
  const [currentPage, setCurrentPage] = useState(1);
  // Track the window width
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  // Update windowWidth when the window is resized
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);

    // Clean up the event listener on unmount
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Determine the number of posts per page based on the current window width
  const postsPerPage =
    windowWidth < 800 ? 3 : windowWidth < 1200 ? 4 : 6;

  const handleCategoryClick = (category) => {
    setChosenCategory(category);
    setCurrentPage(1); // Reset to first page when category changes
  };

  // Filter posts based on the chosen category.
  const filteredPosts =
    chosenCategory === 'All'
      ? blogPosts
      : blogPosts.filter((post) => post.category === chosenCategory);

  // Calculate pagination values
  const totalPosts = filteredPosts.length;
  const totalPages = Math.ceil(totalPosts / postsPerPage);
  const startIndex = (currentPage - 1) * postsPerPage;
  const currentPosts = filteredPosts.slice(startIndex, startIndex + postsPerPage);

  const handlePrevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  const handleNextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
    }
  };

  // Render numbered page buttons
  const renderPageNumbers = () => {
    let pageNumbers = [];
    for (let i = 1; i <= totalPages; i++) {
      pageNumbers.push(
        <button
          key={i}
          className={`pagination-button ${currentPage === i ? 'active' : ''}`}
          onClick={() => setCurrentPage(i)}
        >
          {i}
        </button>
      );
    }
    return pageNumbers;
  };

  return (
    <div className="blog-page">
      <h2>Blog</h2>
      <div className="blog-categories-wrapper">
        <button
          className="cta-button"
          style={{ backgroundColor: chosenCategory === 'All' ? '#325ea8' : '#333' }}
          onClick={() => handleCategoryClick('All')}
        >
          All
        </button>
        <button
          className="cta-button"
          style={{ backgroundColor: chosenCategory === 'Algorithm' ? '#325ea8' : '#333' }}
          onClick={() => handleCategoryClick('Algorithm')}
        >
          Algorithm
        </button>
      </div>
      {/* Pagination Controls */}
      <div className="pagination">
        <button onClick={handlePrevPage} disabled={currentPage === 1}>
          Prev
        </button>
        {renderPageNumbers()}
        <button onClick={handleNextPage} disabled={currentPage === totalPages || totalPages === 0}>
          Next
        </button>
      </div>
      <div className="blog-preview-container">
        {currentPosts.map((post) => (
          <Link to={`/blog/${post.slug}`} key={post.slug} className="blog-preview-link">
            <div className="blog-preview-card">
              <h2>{post.title}</h2>
              <p>{post.excerpt}</p>
              <span>{post.category}</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}

export default BlogPage;
