// src/usePageMeta.js
// Per-page SEO meta + OpenGraph + Twitter + JSON-LD Article schema.
// Drop into any post component: usePageMeta({ title, description, slug, ... }).
import { useEffect } from 'react';

const SITE_URL = 'https://walsertech.ch';
const AUTHOR = 'Bernhard Walser';

/**
 * Set/replace a meta tag. `attr` is 'name' for standard tags, 'property' for og:*.
 */
function setMeta(attr, key, content) {
  if (!content) return;
  let el = document.head.querySelector(`meta[${attr}="${key}"]`);
  if (!el) {
    el = document.createElement('meta');
    el.setAttribute(attr, key);
    document.head.appendChild(el);
  }
  el.setAttribute('content', content);
}

function setCanonical(url) {
  let el = document.head.querySelector('link[rel="canonical"]');
  if (!el) {
    el = document.createElement('link');
    el.setAttribute('rel', 'canonical');
    document.head.appendChild(el);
  }
  el.setAttribute('href', url);
}

function setJsonLd(id, payload) {
  let el = document.head.querySelector(`script[type="application/ld+json"][data-id="${id}"]`);
  if (!el) {
    el = document.createElement('script');
    el.setAttribute('type', 'application/ld+json');
    el.setAttribute('data-id', id);
    document.head.appendChild(el);
  }
  el.textContent = JSON.stringify(payload);
}

/**
 * usePageMeta — sets head tags on mount and restores defaults on unmount.
 *
 * @param {object} opts
 * @param {string} opts.title           Page title (will become `${title} — Bernhard Walser`).
 * @param {string} opts.description     1–2 sentence meta description / OG description.
 * @param {string} opts.slug            Path slug under /blog/, e.g. "self-attention". Omit for the home page.
 * @param {string} [opts.publishedDate] ISO date (YYYY-MM-DD). Optional; powers JSON-LD datePublished.
 * @param {string[]} [opts.keywords]    Optional list of tag keywords.
 * @param {boolean}  [opts.isArticle]   Defaults to true if slug is provided. Set false for non-article pages.
 */
export function usePageMeta(opts) {
  const {
    title,
    description,
    slug,
    publishedDate,
    keywords,
    isArticle = !!slug,
  } = opts || {};

  useEffect(() => {
    if (!title) return;

    const fullTitle = `${title} — ${AUTHOR}`;
    const url = slug ? `${SITE_URL}/blog/${slug}` : SITE_URL;
    const prevTitle = document.title;

    document.title = fullTitle;
    setMeta('name', 'description', description);
    if (keywords && keywords.length) {
      setMeta('name', 'keywords', keywords.join(', '));
    }
    setCanonical(url);

    // OpenGraph (LinkedIn, Facebook, Slack, Discord previews)
    setMeta('property', 'og:title', title);
    setMeta('property', 'og:description', description);
    setMeta('property', 'og:url', url);
    setMeta('property', 'og:site_name', `${AUTHOR}`);
    setMeta('property', 'og:type', isArticle ? 'article' : 'website');
    if (isArticle && publishedDate) {
      setMeta('property', 'article:published_time', publishedDate);
      setMeta('property', 'article:author', AUTHOR);
    }

    // Twitter
    setMeta('name', 'twitter:card', 'summary_large_image');
    setMeta('name', 'twitter:title', title);
    setMeta('name', 'twitter:description', description);

    // JSON-LD Article schema (helps Google's rich-result rendering)
    if (isArticle) {
      setJsonLd('article', {
        '@context': 'https://schema.org',
        '@type': 'Article',
        headline: title,
        description,
        url,
        datePublished: publishedDate || undefined,
        author: { '@type': 'Person', name: AUTHOR, url: SITE_URL },
        publisher: { '@type': 'Person', name: AUTHOR, url: SITE_URL },
        mainEntityOfPage: url,
        keywords: keywords && keywords.length ? keywords.join(', ') : undefined,
      });
    }

    return () => {
      document.title = prevTitle;
      // Remove article-only meta so navigating to the home page doesn't keep them
      const ldEl = document.head.querySelector(`script[data-id="article"]`);
      if (ldEl) ldEl.remove();
    };
  }, [title, description, slug, publishedDate, keywords?.join(','), isArticle]);
}

export default usePageMeta;
