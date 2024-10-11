import React, { useState } from 'react';
import './ContactPage.css';

function ContactPage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const subject = `[Walser Tech Contact] New message from  ${name}`;
    const body = `Name: ${name}\nEmail: ${email}\n\nMessage:\n${message}`;
    window.location.href = `mailto:berniwal1997@hotmail.com?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
  };

  return (
    <div className="contact-page">
      <h2>Contact Us</h2>
      <form className="contact-form" onSubmit={handleSubmit}>
        <label htmlFor="name">Name:</label>
        <input 
          type="text" 
          id="name" 
          name="name" 
          required 
          value={name} 
          onChange={(e) => setName(e.target.value)} 
        />

        <label htmlFor="email">Email:</label>
        <input 
          type="email" 
          id="email" 
          name="email" 
          required 
          value={email} 
          onChange={(e) => setEmail(e.target.value)} 
        />

        <label htmlFor="message">Message:</label>
        <textarea 
          id="message" 
          name="message" 
          rows="4" 
          required 
          value={message} 
          onChange={(e) => setMessage(e.target.value)} 
        ></textarea>

        <button type="submit">Send Message</button>
      </form>
    </div>
  );
}

export default ContactPage;