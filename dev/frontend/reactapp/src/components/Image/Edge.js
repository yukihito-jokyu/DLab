import React from 'react';
import './Edge.css';

function Edge() {
  return (
    <div className="edge-container">
      <div className="element" id="element1">
        Element 1
      </div>
      <div className="edge"></div>
      <div className="element" id="element2">
        Element 2
      </div>
    </div>
  );
}

export default Edge
