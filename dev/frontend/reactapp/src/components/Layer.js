// Layer.jsx
import React, { useState } from 'react';
import LayerStyle from './MiddleLayerStyle';
import InputOutputLayerStyle from './InputOutputLayerStyle';

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
      <h1>入力層</h1>
      <div id='input'>
        <InputOutputLayerStyle />
      </div>
      <h1>中間層</h1>
      <button onClick={increaseElements}>+</button>
      <button onClick={decreaseElements}>-</button>
      
      <div id='structure'>
        {elements.map((element, index) => (
          <div key={index}>{element}</div>
        ))}
      </div>
      <h1>出力層</h1>
      <div id='output'>
        <InputOutputLayerStyle />
      </div>
    </div>
  );
};

export default Layer;
