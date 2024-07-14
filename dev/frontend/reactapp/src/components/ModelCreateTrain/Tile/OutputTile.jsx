import React, { useEffect, useState } from 'react';
import './OutputTile.css';

function OutputTile({ shape }) {
  const [shapeText, setShapeText] = useState('');
  const [errorTile, setErrorTile] = useState(false);
  useEffect(() => {
    const initShape = () => {
      setErrorTile(false);
      if (shape) {
        let text = ''
      shape.forEach((num, index) => {
        if (index > 0) {
          text += ','
        }
        text += num;
        if (num < 1 || num === 'error') {
          setErrorTile(true);
        }
      })
      setShapeText(text);
      }
    };
    initShape();
  }, [shape])
  return (
    <div className='output-tile-wrapper'>
      {errorTile && <div className='error-wrapper'></div>}
      <div className='tile-title-wrapper'>
        <p className='tile-title'>Output</p>
      </div>
      <div className='output-dim-wrapper'>
        <p>{shapeText}</p>
      </div>
    </div>
  )
}

export default OutputTile
