import React from 'react';

/**
 * A simple button component.
 * @param {object} props - The component's props.
 * @param {string} props.label - The text to display on the button.
 * @param {function} props.onClick - The function to call when the button is clicked.
 * @returns {JSX.Element}
 */
function Button({ label, onClick }) {
  return (
    <button onClick={onClick} style={{ padding: '10px', margin: '5px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px' }}>
      {label}
    </button>
  );
}

export default Button;
