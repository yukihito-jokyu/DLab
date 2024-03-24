import React, { useContext } from 'react';
import './InputOutputLayerStyle.css';
import { InputContext } from '../../page/Cartpole';

function InputLayerStyle() {
  // Contextの読み込み
  const { structures } = useContext(InputContext);
  const [ inputNeuron, setInputNeuron ] = structures.inputneuron;

  const handleNeuronChange = (e) => {
    setInputNeuron(parseInt(e.target.value, 10));
  };

  return (
    <div className='layer-wrapper'>
      <div className='neuron'>
        <label htmlFor="LayerStyle">ニューロン数：</label>
        <select id="LayerStyle" value={inputNeuron} onChange={handleNeuronChange}>
          {Array.from({ length: 100 }, (_, index) => index + 1).map((number) => (
            <option key={number} value={number}>{number}</option>
          ))}
        </select>
        <p className='neuron_num'>{inputNeuron}</p>
      </div>
    </div>
  )
}

export default InputLayerStyle;
