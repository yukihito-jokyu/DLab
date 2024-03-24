import React, { useContext } from 'react'
import DnDFild from '../utils/DnDFild';
import MiddleLayerStyle from './MiddleLayerStyle'
import { InputContext } from '../../page/Cartpole';

function MiddleLayer() {
  const { structures } = useContext(InputContext);
  const [middleList, setMiddleList] = structures.middleneuron;
  return (
    <div id='structure'>
      <h1>中間層</h1>
      <DnDFild middleLayer={MiddleLayerStyle} middleData={[middleList, setMiddleList]} />
    </div>
  )
};

export default MiddleLayer;
