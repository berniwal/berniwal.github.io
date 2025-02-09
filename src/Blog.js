// src/Blog.js
import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// You can choose any style you like; here we use 'tomorrow'
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { blogPosts } from './blogPosts';
import './BlogPage.css';

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
          {isCopied ? "Copied!" : "Copy"}
        </button>
        <SyntaxHighlighter
          style={tomorrow}
          language={match[1]}
          PreTag="div"
          customStyle={{
            borderRadius: "8px",
            overflow: "auto",
            padding: "1em"
          }}
          {...props}
        >
          {codeContent}
        </SyntaxHighlighter>
      </div>
    );
  } else {
    return <code className={className} {...props}>{children}</code>;
  }
}

function Blog() {
  const { slug } = useParams();
  const postMeta = blogPosts.find((post) => post.slug === slug);
  const [content, setContent] = useState('');

  useEffect(() => {
    if (postMeta) {
      // Fetch the markdown file from the public/blog folder
      fetch(`/blog/${postMeta.file}`)
        .then((res) => res.text())
        .then((text) => setContent(text))
        .catch((err) => console.error('Error fetching markdown:', err));
    }
  }, [postMeta]);

  if (!postMeta) return <div>Post not found</div>;

  return (
    <div className="main-page">
      <section id="blog-post-section" className="section">
        <div className="blog-post">
          <h1>{postMeta.title}</h1>
          <ReactMarkdown components={{ code: CodeBlock }}>
            {content}
          </ReactMarkdown>
        </div>
      </section>
    </div>
  );
}

export default Blog;
