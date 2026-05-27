// src/Blog.js
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { blogPosts } from './blogPosts';
import VisualizingAttention from './posts/VisualizingAttention';
import VisualizingKVCache from './posts/VisualizingKVCache';
import VisualizingRoPE from './posts/VisualizingRoPE';
import VisualizingRLHF from './posts/VisualizingRLHF';
import VisualizingSelfImprovement from './posts/VisualizingSelfImprovement';
import VisualizingSymbolicRegression from './posts/VisualizingSymbolicRegression';
import MinimaxSearch from './posts/MinimaxSearch';
import AlphaBetaPruning from './posts/AlphaBetaPruning';
import MonteCarloTreeSearch from './posts/MonteCarloTreeSearch';
import './BlogPage.css';

const componentRegistry = {
  VisualizingAttention,
  VisualizingKVCache,
  VisualizingRoPE,
  VisualizingRLHF,
  VisualizingSelfImprovement,
  VisualizingSymbolicRegression,
  MinimaxSearch,
  AlphaBetaPruning,
  MonteCarloTreeSearch,
};

function CodeBlock({ node, inline, className, children, ...props }) {
  const [isCopied, setCopied] = useState(false);
  const codeContent = String(children).replace(/\n$/, '');
  const match = /language-(\w+)/.exec(className || '');

  const handleCopy = () => {
    navigator.clipboard.writeText(codeContent).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  if (!inline && match) {
    return (
      <div className="code-block-wrapper">
        <button onClick={handleCopy} className="copy-button">
          {isCopied ? 'Copied!' : 'Copy'}
        </button>
        <SyntaxHighlighter
          style={oneLight}
          language={match[1]}
          PreTag="div"
          customStyle={{
            borderRadius: 10,
            overflow: 'auto',
            padding: '16px 18px',
            background: '#f1f5f9',
            border: '1px solid #e2e8f0',
            fontSize: 13.5,
            lineHeight: 1.55,
            margin: '24px 0',
          }}
          codeTagProps={{ style: { background: 'transparent', textShadow: 'none' } }}
          {...props}
        >
          {codeContent}
        </SyntaxHighlighter>
      </div>
    );
  }
  return <code className={className} {...props}>{children}</code>;
}

function MarkdownPost({ postMeta }) {
  const [content, setContent] = useState('');
  useEffect(() => {
    fetch(`${process.env.PUBLIC_URL || ''}/blog/${postMeta.file}`)
      .then((res) => res.text())
      .then((text) => setContent(text))
      .catch((err) => console.error('Error fetching markdown:', err));
  }, [postMeta.file]);

  return (
    <article>
      <div className="post-wrap">
        <div className="post-grid">
          <header className="post-header">
            <div className="post-kicker">
              <span>{postMeta.category}</span>
            </div>
            <h1 className="post-title">{postMeta.title}</h1>
            <p className="post-lede">{postMeta.excerpt}</p>
            <div className="post-byline">
              <div className="byline-avatars">
                <div className="byline-avatar">BW</div>
              </div>
              <div className="byline-authors">
                <strong>Bernhard Walser</strong>
              </div>
            </div>
          </header>

          <div className="post-body markdown-body">
            <ReactMarkdown components={{ code: CodeBlock }}>
              {content}
            </ReactMarkdown>
          </div>

          <footer className="post-foot full">
            <div>
              <Link to="/#writing">← Back to writing</Link>
            </div>
          </footer>
        </div>
      </div>
    </article>
  );
}

function Blog() {
  const { slug } = useParams();
  const postMeta = blogPosts.find((post) => post.slug === slug);

  if (!postMeta) return <div className="page" style={{ padding: '96px 28px' }}>Post not found</div>;

  if (postMeta.component) {
    const PostComponent = componentRegistry[postMeta.component];
    if (!PostComponent) return <div className="page" style={{ padding: '96px 28px' }}>Post component not found</div>;
    return <PostComponent />;
  }

  return <MarkdownPost postMeta={postMeta} />;
}

export default Blog;
