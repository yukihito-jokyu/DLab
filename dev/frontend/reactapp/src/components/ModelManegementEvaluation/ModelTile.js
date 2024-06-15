import React, { useState } from 'react';
import './ModelManegementEvaluation.css';
import { ReactComponent as PictureIcon } from '../../assets/svg/graph_24.svg'

function ModelTile({ modelName, accuracy, loss, date, isChecked, modelId, checkBoxChange  }) {
  const [isPicture, setIsPicture] = useState(false);
  const formatTimestamp = (timestamp) => {
    if (timestamp && timestamp.seconds) {
      const date = new Date(timestamp.seconds * 1000);
      return date.toLocaleDateString(); // 日付と時刻をローカル形式で表示
    }
    return '';
  };
  const handleClick = () => {
    setIsPicture(!isPicture);
  };
  return (
    <div className='model-tile-wrapper'>
      <div className='model-title-field'>
        <div className='model-check-box-wrapper'>
          {/* <div className='model-check-box'></div> */}
          <label className="custom-checkbox">
            <input
              type="checkbox"
              checked={isChecked}
              onChange={() => checkBoxChange(modelId)}
            />
            <span className="checkmark"></span>
          </label>
        </div>
        <div className='model-title'>
          <p>{modelName}</p>
        </div>
        <div className='model-accuracy'>
          <p>{accuracy}</p>
        </div>
        <div className='model-loss'>
          <p>{loss}</p>
        </div>
        <div className='model-date'>
          <p>{formatTimestamp(date)}</p>
        </div>
        <div className='model-picture'>
          <div className='model-picture-icon-wrapper' onClick={handleClick}>
            <PictureIcon className='picture-svg' />
          </div>
        </div>
      </div>
      {isPicture && 
        <div className='graph-field'>
          <p>Graph</p>
          <div className='model-picture-filed-wrapper'>
            
          </div>
        </div>
      }
      
    </div>
  )
}

export default ModelTile
