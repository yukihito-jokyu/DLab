import React, { useState } from 'react';
import './ModelCreateTrain.css';
import GradationButton from '../../uiParts/component/GradationButton';
import { useParams } from 'react-router-dom';
import { getModelInput } from '../../db/function/model_structure';

function DataScreen() {
  const { projectName, modelId } = useParams();
  const [i, setI] = useState(0);
  const [normalImage, setNormalImage] = useState(null);
  const [preImage, setPreImage] = useState(null);
  const style1 = {
    width: '200px',
    background: 'linear-gradient(95.34deg, #B6F862 3.35%, #00957A 100%), linear-gradient(94.22deg, #D997FF 0.86%, #50BCFF 105.96%)'
  };
  const handleClick = async () => {
    try {
      const InputInfo = await getModelInput(modelId);
      const data = {
        project_name: projectName,
        input_info: InputInfo
      }
      const response = await fetch('http://127.0.0.1:5000/api/pre_data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      const result = await response.json();
      setNormalImage(result.images);
      setPreImage(result.pre_images);
      console.log(result);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  }
  const increeseI = () => {
    let nextI = i - 1;
    if (nextI === -1) {
      nextI = 9
    }
    setI(nextI)
  }
  const decreeseI = () => {
    let nextI = i + 1;
    if (nextI === 10) {
      nextI = 0
    }
    setI(nextI);
  }
  return (
    <div className='data-screen-wrapper'>
      <div className='image-title-wrapper'>
        <p>Image A</p>
      </div>
      <div className='image-data-wrapper'>
        <div className='image-before-data'>
          {normalImage && <img src={`data:image/png;base64,${normalImage[i]}`} alt='normal_image' />}
        </div>
        <div className='image-arrow'>
          <p>▶</p>
        </div>
        <div className='image-after-data'>
          {preImage && <img src={`data:image/png;base64,${preImage[i]}`} alt='pre_image' />}
        </div>
      </div>
      <div className='image-status-wrapper'>
        <div className='image-before'>
          <p>Before</p>
        </div>
        <div className='image-after'>
          <p>After</p>
        </div>
      </div>
      <div className='change-button-wrapper'>
        <div className='left-button'>
          <div onClick={decreeseI} style={{'cursor': 'pointer'}}>
            <p>◀</p>
          </div>
        </div>
        <div className='right-button'>
          <div onClick={increeseI} style={{'cursor': 'pointer'}}>
            <p>▶</p>
          </div>
        </div>
      </div>
      <div className='confirm-button-wrapper'>
        <div onClick={handleClick}>
          <GradationButton text={'check'} style1={style1} />
        </div>
      </div>
    </div>
  )
}

export default DataScreen
