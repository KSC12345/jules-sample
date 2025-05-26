import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const chatWindowRef = useRef(null);

  // Scroll to bottom of chat window when messages change
  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages]);

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
    setError(null); // Clear error when user starts typing
  };

  const handleSendMessage = async () => {
    if (inputValue.trim() === '') {
      return;
    }

    const userMessage = { text: inputValue, sender: 'user' };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage.text }),
      });

      if (!response.ok) {
        // Try to parse error response from backend if available
        let errorData;
        try {
            errorData = await response.json();
        } catch (e) {
            // If response is not JSON, use status text
            errorData = { detail: response.statusText };
        }
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      
      // The backend response structure is assumed to be:
      // { llm_response: "...", retrieved_context_chunks: ["..."] }
      const botMessage = { 
        text: data.llm_response || "Sorry, I couldn't get a response.", 
        sender: 'bot',
        retrieved_chunks: data.retrieved_context_chunks || [] // Store for potential display
      };
      setMessages(prevMessages => [...prevMessages, botMessage]);

    } catch (err) {
      console.error("Failed to send message:", err);
      const errorMessage = { text: `Error: ${err.message || 'Could not connect to the server.'}`, sender: 'error' };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
      setError(err.message || 'Could not connect to the server.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !isLoading) {
      handleSendMessage();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>RAG Chatbot</h1>
      </header>
      <div className="chat-window" ref={chatWindowRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message-container ${msg.sender}`}>
            <div className={`message ${msg.sender}`}>
              <p>{msg.text}</p>
              {/* Optionally display retrieved chunks for bot messages for debugging/info */}
              {msg.sender === 'bot' && msg.retrieved_chunks && msg.retrieved_chunks.length > 0 && (
                <details className="retrieved-chunks">
                  <summary>Retrieved Context ({msg.retrieved_chunks.length})</summary>
                  <ul>
                    {msg.retrieved_chunks.map((chunk, i) => (
                      <li key={i}>{chunk.length > 100 ? chunk.substring(0, 100) + "..." : chunk}</li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message-container bot">
            <div className="message bot typing-indicator">
              <p>Bot is typing...</p>
            </div>
          </div>
        )}
      </div>
      {error && <p className="error-message">{error}</p>}
      <div className="input-area">
        <input
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button onClick={handleSendMessage} disabled={isLoading}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default App;
