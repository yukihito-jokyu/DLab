// Layer.jsx
import React from 'react';
// import MiddleLayerStyle from './MiddleLayerStyle';
import InputLayerStyle from './InputLayerStyle';
import OutputLayerStyle from './OutputLayerStyle';
import MiddleLayer from './MiddleLayer';

const CartPoleLayer = () => {
  // const [elements, setElements] = useState([]);
  
  // Memoized callback using useCallback
  

  // const increaseElements = () => {
  //   const newElement = <div key={elements.length + 1}><MiddleLayerStyle /></div>;
  //   setElements([...elements, newElement]);
  // };

  // const decreaseElements = () => {
  //   if (elements.length > 0) {
  //     const updatedElements = [...elements];
  //     updatedElements.pop();
  //     setElements(updatedElements);
  //   }
  // };


  return (
    <div>
      <h1>入力層</h1>
      <div id='input'>
        <InputLayerStyle />
      </div>
      {/* <h1>中間層</h1> */}
      {/* <button onClick={increaseElements}>+</button>
      <button onClick={decreaseElements}>-</button>
      
      <div id='structure'>
        {elements.map((element, index) => (
          <div key={index}>{element}</div>
        ))}
      </div> */}
      <MiddleLayer />
      <h1>出力層</h1>
      <div id='output'>
        <OutputLayerStyle />
      </div>
    </div>
  );
};

export default CartPoleLayer;
