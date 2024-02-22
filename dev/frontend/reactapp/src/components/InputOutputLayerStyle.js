import React, { useState } from 'react'

function InputOutputLayerStyle() {
  const [selectedNeuron, setSelectedNeuron] = useState(1);

  const handleNeuronChange = (e) => {
    setSelectedNeuron(parseInt(e.target.value, 10));
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
    </div>
  )
}

export default InputOutputLayerStyle
