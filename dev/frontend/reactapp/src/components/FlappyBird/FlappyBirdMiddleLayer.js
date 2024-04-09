import React, { useContext } from 'react';
import DnDFild from '../utils/DnDFild';
import MiddleLayerStyle from '../CartPole/MiddleLayerStyle';
import { CNNContext } from '../../page/Flappybird';

function FlappyBirdMiddleLayer() {
  const { trainInfo } = useContext(CNNContext);
  const [middleList, setMiddleList] = trainInfo.middleLayer;
  return (
    <div>
      <h1>中間層</h1>
      <DnDFild middleLayer={MiddleLayerStyle} middleData={[middleList, setMiddleList]} />
    </div>
  )
}

export default FlappyBirdMiddleLayer
