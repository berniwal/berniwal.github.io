// src/ConsentBanner.js
// Minimal cookie/analytics consent banner. Pairs with the Consent Mode v2
// setup in public/index.html. On accept, flips analytics_storage to granted
// and fires the first page_view for the currently visible route.
import React, { useState } from 'react';

const STORAGE_KEY = 'analytics-consent';

function readChoice() {
  try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}
function writeChoice(val) {
  try { localStorage.setItem(STORAGE_KEY, val); } catch {}
}

function firePageView() {
  if (typeof window === 'undefined' || !window.gtag) return;
  const path = window.location.pathname + window.location.hash;
  window.gtag('event', 'page_view', {
    page_path: path,
    page_location: window.location.href,
    page_title: document.title,
  });
}

export default function ConsentBanner() {
  const [choice, setChoice] = useState(readChoice);
  if (choice) return null;

  const decide = (val) => {
    writeChoice(val);
    if (val === 'granted' && window.gtag) {
      window.gtag('consent', 'update', { analytics_storage: 'granted' });
      // Send the page_view for the route they're already on
      firePageView();
    }
    setChoice(val);
  };

  return (
    <div className="consent-banner" role="dialog" aria-label="Analytics consent">
      <div className="consent-banner-inner">
        <p>
          This site uses Google Analytics to count pageviews. No ads, no
          cross-site tracking. You can change your mind any time by clearing
          this browser's site data.
        </p>
        <div className="consent-banner-actions">
          <button type="button" className="consent-btn" onClick={() => decide('denied')}>
            Decline
          </button>
          <button type="button" className="consent-btn consent-btn-primary" onClick={() => decide('granted')}>
            Accept
          </button>
        </div>
      </div>
    </div>
  );
}
