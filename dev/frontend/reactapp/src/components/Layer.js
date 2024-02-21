// Layer.jsx
import React, { useState } from 'react';
import LayerStyle from './LayerStyle';

const Layer = () => {
  const [elements, setElements] = useState([]);
  
  // Memoized callback using useCallback
  

  const increaseElements = () => {
    const newElement = <div key={elements.length + 1}><LayerStyle /></div>;
    setElements([...elements, newElement]);
  };

  const decreaseElements = () => {
    if (elements.length > 0) {
      const updatedElements = [...elements];
      updatedElements.pop();
      setElements(updatedElements);
    }
  };


  return (
    <div>
      <button onClick={increaseElements}>+</button>
      <button onClick={decreaseElements}>-</button>
      
      <div id='structure'>
        {elements.map((element, index) => (
          <div key={index}>{element}</div>
        ))}
      </div>
    </div>
  );
};

export default Layer;
