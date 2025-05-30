import React from 'react';

/**
 * A simple card component.
 * @param {object} props - The component's props.
 * @param {string} props.title - The title of the card.
 * @param {React.ReactNode} props.children - The content of the card.
 * @returns {JSX.Element}
 */
function Card({ title, children }) {
  return (
    <div style={{ border: '1px solid #ccc', borderRadius: '8px', padding: '16px', margin: '10px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
      {title && <h3 style={{ marginTop: 0 }}>{title}</h3>}
      <div>{children}</div>
    </div>
  );
}

export default Card;
