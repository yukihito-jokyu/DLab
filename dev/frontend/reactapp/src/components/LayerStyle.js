import React, { useState } from 'react';
import './LayerStyle.css';

function LayerStyle() {
  const [selectedNeuron, setSelectedNeuron] = useState(1);
  const [selectedActivation, setSelectedActivation] = useState('ReLU');

  const handleNeuronChange = (e) => {
    setSelectedNeuron(parseInt(e.target.value, 10));
  };

  const handleActivationChange = (e) => {
    setSelectedActivation(e.target.value);
  };


  return (
    <div className='layer-wrapper'>
      <div className='neuron'>
        <label htmlFor="LayerStyle">ニューロン数：</label>
        <select id="LayerStyle" value={selectedNeuron} onChange={handleNeuronChange}>
          {Array.from({ length: 100 }, (_, index) => index + 1).map((number) => (
            <option key={number} value={number}>{number}</option>
          ))}
        </select>
        <p className='neuron_num'>{selectedNeuron}</p>
      </div>
      <div className='activ-func'>
        <label htmlFor="activFuncSelector">活性化関数：</label>
        <select id="activFuncSelector" value={selectedActivation} onChange={handleActivationChange}>
          <option value="ReLU">ReLU</option>
          <option value="Sigmoid">Sigmoid</option>
          <option value="Tanh">Tanh</option>
          <option value="Softmax">Softmax</option>
          <option value="None">None</option>
        </select>
        <p className='neuron_activ'>{selectedActivation}</p>
      </div>
    </div>
  );
}

export default LayerStyle;
