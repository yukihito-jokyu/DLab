import React, { useContext } from 'react';
import '../CartPole/InputOutputLayerStyle.css';
import { CNNContext } from '../../page/Flappybird';

function FlappyBirdOutputLayerStyle() {
  // Contextの読み込み
  const { trainInfo } = useContext(CNNContext);
  const [ outputNeuron, setOutputNeuron ] = trainInfo.outputLayer;

  const handleNeuronChange = (e) => {
    setOutputNeuron(parseInt(e.target.value, 10));
  };

  return (
    <div className='layer-wrapper'>
      <div className='neuron'>
        <label htmlFor="LayerStyle">ニューロン数：</label>
        <select id="LayerStyle" value={outputNeuron} onChange={handleNeuronChange}>
          {Array.from({ length: 100 }, (_, index) => index + 1).map((number) => (
            <option key={number} value={number}>{number}</option>
          ))}
        </select>
        <p className='neuron_num'>{outputNeuron}</p>
      </div>
    </div>
  )
}

export default FlappyBirdOutputLayerStyle;
