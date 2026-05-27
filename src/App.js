// src/App.js
import React, { useEffect, useState } from 'react';
import { HashRouter as Router, Routes, Route, useLocation, Link } from 'react-router-dom';
import { HashLink } from 'react-router-hash-link';
import './App.css';
import MainPage from './MainPage';
import Blog from './Blog';

function ScrollToTopOnRoute() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
  }, [pathname]);
  return null;
}

// Global scroll progress bar — only on long pages (i.e. anywhere there's
// meaningful scroll headroom). Sits above the nav.
function ReadingProgress() {
  const [pct, setPct] = useState(0);
  const { pathname } = useLocation();
  useEffect(() => {
    const onScroll = () => {
      const h = document.documentElement;
      const total = h.scrollHeight - h.clientHeight;
      setPct(total > 0 ? Math.max(0, Math.min(1, h.scrollTop / total)) : 0);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, [pathname]);
  return (
    <div className="reading-progress" aria-hidden="true">
      <div className="reading-progress-bar" style={{ width: `${(pct * 100).toFixed(2)}%` }} />
    </div>
  );
}

// Cross-route reveal observer — watches all .reveal / .reveal-stagger nodes
// on the page and adds .in when they intersect the viewport. Re-runs on
// every route change so newly mounted elements get picked up.
function useRevealObserver(pathname) {
  useEffect(() => {
    // Give React a tick to mount the new route's DOM
    const id = setTimeout(() => {
      if (typeof IntersectionObserver === 'undefined') {
        document.querySelectorAll('.reveal, .reveal-stagger').forEach((el) => el.classList.add('in'));
        return;
      }
      const io = new IntersectionObserver(
        (entries) => {
          entries.forEach((e) => {
            if (e.isIntersecting) {
              e.target.classList.add('in');
              io.unobserve(e.target);
            }
          });
        },
        { rootMargin: '0px 0px -10% 0px', threshold: 0.05 },
      );
      document
        .querySelectorAll('.reveal:not(.in), .reveal-stagger:not(.in), .post-2026 .viz-panel:not(.in), .post-2026 .post-code-details:not(.in)')
        .forEach((el) => io.observe(el));
      // Stash so we can disconnect on next route
      if (window.__revealIO) window.__revealIO.disconnect();
      window.__revealIO = io;
    }, 40);
    return () => clearTimeout(id);
  }, [pathname]);
}

function RouteEffects() {
  const { pathname } = useLocation();
  useRevealObserver(pathname);
  return null;
}

function Nav() {
  return (
    <nav className="nav">
      <div className="nav-inner">
        <Link to="/" className="nav-brand">
          <span className="nav-mark">BW</span>
          <span>Bernhard Walser</span>
        </Link>
        <div className="nav-links">
          <HashLink smooth to="/#writing">Writing</HashLink>
          <HashLink smooth to="/#experience">Experience</HashLink>
          <HashLink smooth to="/#contact">Contact</HashLink>
        </div>
      </div>
    </nav>
  );
}

function SiteFooter() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div>© {new Date().getFullYear()} Bernhard Walser · Built in Zurich</div>
        <div className="footer-links">
          <HashLink smooth to="/#writing">Writing</HashLink>
          <HashLink smooth to="/#experience">Experience</HashLink>
          <HashLink smooth to="/#contact">Contact</HashLink>
        </div>
      </div>
    </footer>
  );
}

function App() {
  return (
    <Router>
      <ScrollToTopOnRoute />
      <RouteEffects />
      <ReadingProgress />
      <div className="app-container">
        <Nav />
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/blog/:slug" element={<Blog />} />
        </Routes>
        <SiteFooter />
      </div>
    </Router>
  );
}

export default App;
